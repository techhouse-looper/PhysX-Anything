#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

VLM_ENV = "physx-anything"
DECODER_ENV = "physx-decoder"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "pipeline_runs"
MAX_NEW_TOKENS = "16384"


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem) or "image"


def run_key(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    suffix = path.suffix.lower().lstrip(".") or "file"
    return f"{safe_stem(path)}_{suffix}_{digest}"


def run_step(name: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PhysX-Anything VLM→decoder→split→sim-ready for one image."
    )
    parser.add_argument("image_path", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    run_name = run_key(image_path)
    run_dir = RUN_ROOT / run_name
    demo_dir = run_dir / "demo"
    output_dir = run_dir / "test_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged_image = demo_dir / f"{run_name}{image_path.suffix.lower()}"
    shutil.copy2(image_path, staged_image)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    run_step(
        "1_vlm_demo",
        [
            "conda", "run", "-n", VLM_ENV, "python", "1_vlm_demo.py",
            "--quant_config", "configs/inference/vlm_4bit_quantization.yaml",
            "--demo_path", str(demo_dir),
            "--output_path", str(output_dir),
            "--limit", "1",
            "--max_new_tokens", MAX_NEW_TOKENS,
            "--save_part_ply", "False",
            "--remove_bg", "True",
        ],
        env=env,
    )
    run_step(
        "2_decoder",
        [
            "conda", "run", "-n", DECODER_ENV, "python", "2_decoder.py",
            "--demo_path", str(demo_dir),
            "--output_path", str(output_dir),
            "--decoder_ckpt", "./pretrain/decoder",
            "--limit", "1",
            "--low_vram", "True",
        ],
        env=env,
    )
    run_step(
        "3_split",
        [
            "conda", "run", "-n", DECODER_ENV, "python", "3_split.py",
            "--basepath", str(output_dir),
            "--index", "0",
            "--range", "2000",
        ],
        env=env,
    )
    run_step(
        "4_simready_gen",
        [
            "conda", "run", "-n", DECODER_ENV, "python", "4_simready_gen.py",
            "--voxel_define", "32",
            "--basepath", str(output_dir),
            "--process", "0",
            "--fixed_base", "0",
            "--deformable", "0",
        ],
        env=env,
    )

    result_dir = output_dir / run_name
    print("\n=== complete ===")
    print(f"Result directory: {result_dir}")
    for path in ["basic_info.txt", "sample.glb", "basic_info.json", "basic.urdf", "basic.xml"]:
        candidate = result_dir / path
        print(f"{path}: {'OK' if candidate.exists() else 'missing'}")


if __name__ == "__main__":
    main()
