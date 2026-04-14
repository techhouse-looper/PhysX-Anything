Decoder import probe after successful quantized VLM smoke test (2026-04-14)

Command:
conda run -n physx-anything python - <<'PY'
from trellis.pipelines import TrellisImageTo3DPipeline
PY

Resolved missing packages iteratively:
- easydict
- ipdb (needed by TRELLIS vendored modules)
- plyfile
- utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

Current blocker:
- ModuleNotFoundError: No module named 'kaolin'

Notes:
- kaolin is distributed via NVIDIA's PyTorch/CUDA-specific wheel indexes, not as a normal PyPI package for this use case.
- Current working VLM env: PyTorch 2.4.0 + CUDA 11.8 runtime.
- setup.sh's kaolin path lists torch 2.4.0 under a cu121 wheel index, so do not blindly install kaolin into the working VLM env without deciding whether to migrate the decoder env to CUDA 12.1 or create a separate decoder env.
- Keep VLM 4-bit evidence intact before changing torch/CUDA stack.
