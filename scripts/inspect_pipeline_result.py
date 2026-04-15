#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FILES = ["basic_info.txt", "basic_info.json", "sample.glb", "basic.urdf", "basic.xml"]
COORD_TOKEN_RE = re.compile(r"^\d+(?:-\d+)?$")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def add_warning(report: dict[str, Any], message: str) -> None:
    report["warnings"].append(message)


def add_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def file_info(path: Path) -> dict[str, Any]:
    return {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def check_required_files(result_dir: Path, report: dict[str, Any]) -> None:
    for name in REQUIRED_FILES:
        info = file_info(result_dir / name)
        report["required_files"][name] = info
        if not info["exists"] or info["bytes"] == 0:
            add_error(report, f"missing or empty required file: {name}")

    objs_dir = result_dir / "objs"
    report["required_files"]["objs/"] = {"exists": objs_dir.is_dir(), "bytes": None}
    if not objs_dir.is_dir():
        add_error(report, "missing required directory: objs/")

    if report["required_files"].get("sample.glb", {}).get("bytes", 0) < 100_000:
        add_warning(report, "sample.glb is very small; generated mesh may be empty or incomplete")
    if report["required_files"].get("basic.xml", {}).get("bytes", 0) < 500:
        add_warning(report, "basic.xml is very small; sim-ready XML may be incomplete")


def check_basic_info(result_dir: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    path = result_dir / "basic_info.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        add_error(report, f"basic_info.json parse failed: {exc}")
        return None

    parts = data.get("parts", [])
    group_info = data.get("group_info", {})
    report["basic_info"] = {
        "object_name": data.get("object_name"),
        "category": data.get("category"),
        "dimension": data.get("dimension"),
        "part_count": len(parts),
        "group_count": len(group_info) if isinstance(group_info, dict) else None,
        "parts": [
            {
                "label": part.get("label"),
                "name": part.get("name"),
                "material": part.get("material"),
                "priority_rank": part.get("priority_rank"),
            }
            for part in parts
            if isinstance(part, dict)
        ],
    }

    if not parts:
        add_warning(report, "basic_info.json contains no parts")
    if isinstance(group_info, dict):
        for group_id, group in group_info.items():
            if isinstance(group, list) and len(group) >= 4 and group[-1] != "E":
                params = group[2]
                if isinstance(params, list) and len(params) >= 6:
                    axis_pos = params[3:6]
                    if any(v < -0.55 or v > 0.55 for v in axis_pos):
                        add_warning(report, f"joint axis for group {group_id} is near/outside normalized bounds: {axis_pos}")
    return data


def check_coord_text(result_dir: Path, report: dict[str, Any]) -> None:
    for path in sorted(result_dir.glob("coord_*.txt")):
        text = path.read_text(errors="ignore").strip()
        tokens = text.split()
        malformed = []
        descending = []
        for token in tokens:
            if not COORD_TOKEN_RE.match(token):
                malformed.append(token)
                continue
            if "-" in token:
                a, b = map(int, token.split("-"))
                if a > b:
                    descending.append(token)
        item = {"file": path.name, "bytes": path.stat().st_size, "tokens": len(tokens)}
        report["coord_text"].append(item)
        if malformed:
            add_warning(report, f"{path.name} has malformed coordinate tokens: {malformed[:5]}")
        if descending:
            add_warning(report, f"{path.name} has descending coordinate ranges: {descending[:5]}")
        if path.stat().st_size < 50:
            add_warning(report, f"{path.name} is very small; part may be under-generated")


def check_voxel_arrays(result_dir: Path, report: dict[str, Any]) -> None:
    total_ind_voxels = 0
    for path in sorted(result_dir.glob("ind_*.npy")):
        try:
            arr = np.load(path)
        except Exception as exc:
            add_error(report, f"failed to load {path.name}: {exc}")
            continue
        item = {"file": path.name, "shape": list(arr.shape), "voxels": int(arr.shape[0]) if arr.ndim else 0}
        if arr.size and arr.ndim == 2 and arr.shape[1] == 3:
            item["min"] = arr.min(axis=0).astype(int).tolist()
            item["max"] = arr.max(axis=0).astype(int).tolist()
        report["voxel_parts"].append(item)
        if arr.ndim != 2 or arr.shape[1] != 3:
            add_error(report, f"{path.name} has invalid shape: {arr.shape}")
            continue
        total_ind_voxels += arr.shape[0]
        if arr.shape[0] < 20:
            add_warning(report, f"{path.name} has very few voxels: {arr.shape[0]}")
        if arr.size and ((arr < 0).any() or (arr > 31).any()):
            add_error(report, f"{path.name} has out-of-range voxel coordinates")

    allind = result_dir / "allind.npy"
    if allind.exists():
        arr = np.load(allind)
        report["allind"] = {"shape": list(arr.shape), "voxels": int(arr.shape[0]) if arr.ndim else 0}
        if arr.ndim != 2 or arr.shape[1] != 3:
            add_error(report, f"allind.npy has invalid shape: {arr.shape}")
        elif total_ind_voxels and arr.shape[0] != total_ind_voxels:
            add_warning(report, f"allind voxel count {arr.shape[0]} differs from sum(ind_*.npy) {total_ind_voxels}")


def count_obj_mesh(path: Path) -> dict[str, Any]:
    vertices = 0
    faces = 0
    with path.open("r", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return {"file": str(path), "bytes": path.stat().st_size, "vertices": vertices, "faces": faces}


def check_obj_parts(result_dir: Path, report: dict[str, Any]) -> None:
    objs_dir = result_dir / "objs"
    for obj in sorted(objs_dir.glob("*/*.obj")):
        item = count_obj_mesh(obj)
        item["file"] = rel(obj, result_dir)
        report["obj_parts"].append(item)
        if item["vertices"] == 0 or item["faces"] == 0:
            add_error(report, f"{item['file']} has empty mesh")
        elif item["vertices"] < 100 or item["faces"] < 100:
            add_warning(report, f"{item['file']} is very small: V={item['vertices']} F={item['faces']}")
    if objs_dir.exists() and not report["obj_parts"]:
        add_error(report, "objs/ exists but no part OBJ files were found")


def check_glb(result_dir: Path, report: dict[str, Any]) -> None:
    path = result_dir / "sample.glb"
    if not path.exists():
        return
    try:
        import trimesh
        loaded = trimesh.load(path)
        info = {"load": "ok"}
        if hasattr(loaded, "geometry"):
            info["geometry_count"] = len(loaded.geometry)
        if hasattr(loaded, "bounds") and loaded.bounds is not None:
            info["bounds"] = np.asarray(loaded.bounds).tolist()
        report["glb"] = info
    except Exception as exc:
        report["glb"] = {"load": "fail", "error": str(exc)}
        add_error(report, f"sample.glb load failed: {exc}")


def check_mujoco(result_dir: Path, report: dict[str, Any]) -> None:
    try:
        import mujoco
    except ImportError:
        report["mujoco"] = {"load": "skipped_missing_package"}
        add_warning(report, "mujoco is not installed; basic.xml load check skipped")
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(result_dir / "basic.xml"))
        report["mujoco"] = {"load": "ok", "nbody": int(model.nbody), "njnt": int(model.njnt), "ngeom": int(model.ngeom)}
    except Exception as exc:
        report["mujoco"] = {"load": "fail", "error": str(exc)}
        add_error(report, f"MuJoCo basic.xml load failed: {exc}")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Inspection report: {Path(report['result_dir']).name}",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Errors",
        "",
    ]
    lines += [f"- {item}" for item in report["errors"]] or ["- None"]
    lines += ["", "## Warnings", ""]
    lines += [f"- {item}" for item in report["warnings"]] or ["- None"]
    lines += ["", "## Required files", ""]
    for name, info in report["required_files"].items():
        lines.append(f"- `{name}`: {info}")
    lines += ["", "## Object parts", ""]
    for item in report.get("obj_parts", []):
        lines.append(f"- `{item['file']}`: V={item['vertices']} F={item['faces']} bytes={item['bytes']}")
    path.write_text("\n".join(lines) + "\n")


def inspect(result_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "result_dir": str(result_dir),
        "required_files": {},
        "warnings": [],
        "errors": [],
        "basic_info": {},
        "coord_text": [],
        "voxel_parts": [],
        "allind": {},
        "obj_parts": [],
        "glb": {},
        "mujoco": {},
    }
    check_required_files(result_dir, report)
    check_basic_info(result_dir, report)
    check_coord_text(result_dir, report)
    check_voxel_arrays(result_dir, report)
    check_obj_parts(result_dir, report)
    check_glb(result_dir, report)
    check_mujoco(result_dir, report)
    report["status"] = "FAIL" if report["errors"] else "PASS_WITH_WARNINGS" if report["warnings"] else "PASS"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a PhysX-Anything pipeline result directory.")
    parser.add_argument("result_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")
    report = inspect(result_dir)
    summary_path = result_dir / "inspection_summary.json"
    report_path = result_dir / "inspection_report.md"
    summary_path.write_text(json.dumps(report, indent=2))
    write_markdown(report, report_path)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
