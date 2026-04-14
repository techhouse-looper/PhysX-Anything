from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import base64
import os
import ipdb
import numpy as np
from PIL import Image
import trimesh
from rembg import remove
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def torch_dtype_from_name(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_quantization_config(args):
    if args.quantization == "none":
        return None

    from transformers import BitsAndBytesConfig

    if args.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype_from_name(args.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    if args.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(f"Unsupported quantization mode: {args.quantization}")


def load_vlm_model(args):
    dtype = torch_dtype_from_name(args.torch_dtype)
    quantization_config = build_quantization_config(args)
    attn_candidates = (
        ["flash_attention_2", "sdpa"]
        if args.attn_implementation == "auto"
        else [args.attn_implementation]
    )

    last_error = None
    for attn_implementation in attn_candidates:
        kwargs = {
            "torch_dtype": dtype,
            "device_map": args.device_map,
        }
        if attn_implementation != "none":
            kwargs["attn_implementation"] = attn_implementation
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        try:
            print(
                f"[load] ckpt={args.ckpt} quantization={args.quantization} "
                f"dtype={args.torch_dtype} attn={attn_implementation} device_map={args.device_map}"
            )
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(args.ckpt, **kwargs)
        except Exception as exc:
            last_error = exc
            if args.attn_implementation == "auto" and attn_implementation == "flash_attention_2":
                print(f"[warn] flash_attention_2 load failed; falling back to sdpa: {exc}")
                continue
            raise

    raise RuntimeError("Failed to load VLM model") from last_error
def voxel_encode(voxels: np.ndarray, size: int = 32) -> np.ndarray:

    voxels = np.asarray(voxels, dtype=np.int64)
    assert voxels.ndim == 2 and voxels.shape[1] == 3, "voxels shape should be (N,3)"
    assert size == 32, "size=32（2^5）。"
    if (voxels < 0).any() or (voxels >= size).any():
        raise ValueError("xyz should be within [0, 32).")

    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    return (x << 10) | (y << 5) | z


def voxel_decode(indices: np.ndarray, size: int = 32) -> np.ndarray:

    indices = np.asarray(indices, dtype=np.int64).ravel()
    assert size == 32, "size=32（2^5）。"
    if (indices < 0).any() or (indices >= size**3).any():

        indices=indices.clip(0,size**3-1)
        print("index should be within [0, 32768).")


    x = (indices >> 10) & 31
    y = (indices >> 5)  & 31
    z = indices & 31
    return np.stack([x, y, z], axis=1)



def ints_to_space_separated_str(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return " ".join(map(str, arr))



def merge_adjacent_to_dash(s: str) -> str:

    if not s.strip():
        return ""

    nums = list(map(int, s.split()))

    nums = sorted(set(nums))

    result = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            result.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    result.append(f"{start}-{prev}" if start != prev else f"{start}")
    return " ".join(result)



def dash_str_to_ints(s: str) -> np.ndarray:

    if not s.strip():
        return np.array([], dtype=np.int64)

    out = []
    for token in s.split():
        if "-" in token:
            a, b = map(int, token.split("-"))
            if a > b:
                a, b = b, a 
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    return np.array(sorted(set(out)), dtype=np.int64)


def addmessage(message,before,after):
    answer={}
    answer['role']='assistant'
    answer['content']=[{"type": "text", "text": before}]
    question={}
    question['role']='user'
    question['content']=[{"type": "text", "text": after}]
    newmessage=message.copy()
    newmessage.append(answer)
    newmessage.append(question)
    return newmessage



def generate_save(model,messages,save_dir,save_name='test',save=True,max_length=32768,max_new_tokens=None):


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)


    generation_kwargs = {
        "do_sample": False,
        "temperature": 0,
    }
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens
    else:
        generation_kwargs["max_length"] = max_length

    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if save:
        with open(os.path.join(save_dir,save_name+'.txt'),'w') as file:
            file.write( output_text[0])
    return output_text[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, default='./demo')
    parser.add_argument("--output_path", type=str, default='./test_demo')
    parser.add_argument("--save_part_ply", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--remove_bg", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--ckpt", type=str, default='./pretrain/vlm')
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--bnb_4bit_compute_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--bnb_4bit_quant_type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", choices=["auto", "flash_attention_2", "sdpa", "eager", "none"], default="auto")
    parser.add_argument("--min_pixels", type=int, default=65536)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    basepath=args.demo_path
    namelist=sorted(
        name for name in os.listdir(basepath)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    )
    if args.limit is not None:
        namelist = namelist[:args.limit]
    


    model = load_vlm_model(args)
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.image_processor.min_pixels=min_pixels
    processor.image_processor.max_pixels=max_pixels
    processor.image_processor.size["shortest_edge"]=min_pixels
    processor.image_processor.size["longest_edge"]=max_pixels

    for name in namelist:



        save_dir=os.path.join(args.output_path, os.path.splitext(name)[0])
        os.makedirs(os.path.join(save_dir), exist_ok=True)

        image_path = os.path.join(basepath,name)



        with open(os.path.join('./dataset/overall_prompt.txt'), "r", encoding="utf-8") as f:
            basicqu = f.read()

        input_image = Image.open(image_path)
        im_resized = input_image.resize((512, 512), Image.LANCZOS)

        if args.remove_bg:
            im_resized = remove(im_resized)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": im_resized.convert("RGB"),
                    },
                    {"type": "text", "text": basicqu},
                ],
            }
        ]
        
    

        basicoutput=generate_save(
            model,
            messages,
            save_dir,
            'basic_info',
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        index=0
        while 'l_'+str(index) in basicoutput:
            index+=1

        allcoord=[]
        for part in range(index):

            question="Based on the structured description of l_"+str(part)+", generate its 3D voxel grid in the following format (voxel grid=32, use numbers from 0 to 32767, merge maximal consecutive runs: 199...216 -> 199-216): 184 198 199-216 230-237..."
            messages1=addmessage(messages,basicoutput,question)
            output1=generate_save(
                model,
                messages1,
                save_dir,
                'coord_'+str(part),
                save=True,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )
            print(len(messages1))
            idx_back = dash_str_to_ints(output1)
            voxels_back = voxel_decode(idx_back)
            allcoord.append(voxels_back)
            np.save(os.path.join(save_dir,'ind_'+str(part)+'.npy'),voxels_back)
            if args.save_part_ply:
                partply=trimesh.points.PointCloud(voxels_back)
                partply.export(os.path.join(save_dir,'ind_'+str(part)+'.ply'))

        np.save(os.path.join(save_dir,'allind.npy'),np.concatenate(allcoord))
