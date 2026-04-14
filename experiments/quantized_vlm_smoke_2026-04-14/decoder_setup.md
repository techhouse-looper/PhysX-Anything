# Decoder setup notes — 2026-04-15

This note records the decoder environment used after the quantized VLM smoke test produced `test_demo/0/allind.npy`.

## Environment strategy

Use a separate decoder env so the verified VLM quantization env stays stable:

```bash
conda create -n physx-decoder python=3.10 -y
conda run -n physx-decoder python -m pip install --upgrade pip
conda run -n physx-decoder python -m pip install \
  torch==2.1.1+cu118 torchvision==0.16.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n physx-decoder python -m pip install \
  xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
conda run -n physx-decoder python -m pip install \
  spconv-cu118 kaolin==0.15.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html
conda run -n physx-decoder python -m pip install \
  numpy==1.24.3 'opencv-python-headless<4.12' \
  pillow imageio imageio-ffmpeg tqdm easydict scipy ninja rembg onnxruntime \
  trimesh open3d xatlas pyvista pymeshfix igraph transformers==4.50.0 \
  safetensors huggingface_hub==0.36.2 plyfile ipdb \
  'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'
conda run -n physx-decoder python -m pip install \
  typing-extensions packaging filelock tomli platformdirs 'setuptools<81' wheel
conda install -n physx-decoder -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 -y
CUDA_HOME="$CONDA_PREFIX" conda run -n physx-decoder python -m pip install \
  --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
conda run -n physx-decoder python -m pip install \
  --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

Important compatibility notes:

- `kaolin==0.15.0` worked with the NVIDIA wheel index for `torch-2.1.1_cu118`.
- `numpy==1.24.3` is required; NumPy 2.x caused a Kaolin binary ABI error.
- `setuptools<81` is required because `torch.utils.cpp_extension` in torch 2.1.1 still imports `pkg_resources`.
- `nvdiffrast` must be built with CUDA 11.8 nvcc from the decoder env, not a newer system CUDA toolkit.
- `ATTN_BACKEND=xformers` avoids needing flash-attn for the decoder smoke test.

## Decoder smoke command

```bash
PYTHONPATH=. conda run -n physx-decoder python 2_decoder.py --limit 1
```

Result:

- Exit status: `0`
- Peak VRAM observed: about `9763 MiB`
- Output: `test_demo/0/sample.glb` (`5049244` bytes in this run)

## Downstream split and sim-ready validation

After `sample.glb` was generated, the existing downstream scripts completed on the same `test_demo/0` asset:

```bash
conda run -n physx-decoder python 3_split.py --index 0 --range 2000
conda run -n physx-decoder python 4_simready_gen.py   --voxel_define 32   --basepath ./test_demo   --process 0   --fixed_base 0   --deformable 0
```

Observed outputs:

- `test_demo/0/objs/0/0.obj`
- `test_demo/0/objs/1/1.obj`
- `test_demo/0/basic_info.json`
- `test_demo/0/basic.urdf`
- `test_demo/0/basic.xml`
- `test_demo/0/desert.png`

## Remaining caveat

The first import probe still prints a `utils3d.__version__` warning when importing `utils3d` directly, but TRELLIS pipeline import and the patched decoder smoke test both complete. Treat this as non-blocking unless a future code path imports `utils3d` as a top-level module differently.
