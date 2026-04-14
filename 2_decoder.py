import argparse
import gc
import os

# Prefer installable wheels over building flash-attn for local decoder smoke tests.
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ["ATTN_BACKEND"])
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
from PIL import Image
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_pipeline(decoder_ckpt):
    pipeline = TrellisImageTo3DPipeline.from_pretrained(decoder_ckpt)
    pipeline.cuda()
    return pipeline


def offload_before_decode(pipeline):
    for key in [
        'image_cond_model',
        'sparse_structure_encoder',
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'slat_flow_model',
        'slat_decoder_rf',
    ]:
        if key in pipeline.models:
            pipeline.models[key].cpu()
    torch.cuda.empty_cache()


@torch.no_grad()
def run_control_low_vram(pipeline, pointinput, image, seed=1, formats=('mesh', 'gaussian')):
    image = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([image])
    torch.manual_seed(seed)

    low = pipeline.models['sparse_structure_encoder'](pointinput, sample_posterior=False)
    coords = pipeline.sample_sparse_structure_control(low, cond, 1, {})
    slat = pipeline.sample_slat(cond, coords, {})

    del low, coords, cond
    offload_before_decode(pipeline)
    return pipeline.decode_slat(slat, list(formats))


def voxel_grid_from_coords(coords, size=32, resolution=64):
    coords = coords + 32 - (size) // 2
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss.cuda().float().unsqueeze(0)


def export_glb(outputs, output_path):
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.5,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(output_path)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def iter_image_names(demo_path, output_path, limit=None):
    names = sorted(
        name for name in os.listdir(demo_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    )
    ready = [
        name for name in names
        if os.path.exists(os.path.join(output_path, os.path.splitext(name)[0], 'allind.npy'))
    ]
    return ready[:limit] if limit is not None else ready


def process_one(pipeline, name, args):
    image = Image.open(os.path.join(args.demo_path, name))
    qwenpath = os.path.join(args.output_path, os.path.splitext(name)[0])
    newcoords = np.load(os.path.join(qwenpath, 'allind.npy'))
    ss = voxel_grid_from_coords(newcoords, size=args.voxel_size, resolution=args.resolution)

    if args.low_vram:
        outputs = run_control_low_vram(pipeline, ss, image, seed=args.seed)
    else:
        outputs = pipeline.run_control(ss, image, seed=args.seed, formats=['mesh', 'gaussian'])

    export_glb(outputs, os.path.join(qwenpath, 'sample.glb'))
    del outputs, ss
    clear_cuda()


def build_parser():
    parser = argparse.ArgumentParser(description="Run PhysX-Anything decoder inference.")
    parser.add_argument("--demo_path", type=str, default='./demo')
    parser.add_argument("--output_path", type=str, default='./test_demo')
    parser.add_argument("--decoder_ckpt", type=str, default='./pretrain/decoder')
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--low_vram", type=str2bool, nargs="?", const=True, default=True)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    namelist = iter_image_names(args.demo_path, args.output_path, limit=args.limit)

    if args.low_vram:
        for name in namelist:
            pipeline = load_pipeline(args.decoder_ckpt)
            try:
                process_one(pipeline, name, args)
            finally:
                del pipeline
                clear_cuda()
    else:
        pipeline = load_pipeline(args.decoder_ckpt)
        try:
            for name in namelist:
                process_one(pipeline, name, args)
        finally:
            del pipeline
            clear_cuda()
