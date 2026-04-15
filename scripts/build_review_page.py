#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path

REQUIRED_ARTIFACTS = ["basic_info.txt", "sample.glb", "basic_info.json", "basic.urdf", "basic.xml"]
CHECKLIST = [
    ("object_identity", "Generated object matches the input category"),
    ("global_geometry", "Global shape and proportions are plausible"),
    ("back_side", "Back/side geometry is not obviously missing or collapsed"),
    ("holes_openings", "Expected holes/openings are preserved"),
    ("major_parts", "Major semantic parts are present"),
    ("part_split", "Part split is reasonable"),
    ("joint_type", "Joint type is plausible"),
    ("joint_axis", "Joint axis and range are plausible"),
    ("simulation_ready", "URDF/XML outputs are suitable for simulation follow-up"),
]
SCORES = ["object_identity", "global_shape", "part_completeness", "holes_openings", "articulation", "simulation_readiness"]
VERDICTS = ["PASS", "PASS_WITH_WARNINGS", "NEEDS_MANUAL_EDIT", "FAIL_RERUN_WITH_BETTER_IMAGE", "FAIL_PIPELINE"]


def read_text(path: Path, limit: int = 6000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    return text if len(text) <= limit else text[:limit] + "\n... truncated ..."


def find_input_images(result_dir: Path) -> list[Path]:
    # Expected wrapper layout: pipeline_runs/<run>/test_demo/<object>.
    run_root = result_dir.parents[1] if len(result_dir.parents) >= 2 else result_dir.parent
    demo_dir = run_root / "demo"
    if not demo_dir.exists():
        return []
    return sorted(path for path in demo_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def rel_link(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_summary(result_dir: Path) -> dict:
    path = result_dir / "inspection_summary.json"
    if not path.exists():
        return {"status": "MISSING_INSPECTION", "warnings": ["inspection_summary.json not found"], "errors": []}
    return json.loads(path.read_text())


def artifact_list(result_dir: Path) -> str:
    items = []
    for name in REQUIRED_ARTIFACTS:
        path = result_dir / name
        if path.exists():
            items.append(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a> ({path.stat().st_size} bytes)</li>')
        else:
            items.append(f'<li class="missing">{html.escape(name)} missing</li>')
    return "\n".join(items)


def checklist_html() -> str:
    return "\n".join(
        f'<label><input type="checkbox" data-key="{key}"> {html.escape(label)}</label>'
        for key, label in CHECKLIST
    )


def scores_html() -> str:
    rows = []
    for key in SCORES:
        options = "".join(f'<option value="{value}">{value}</option>' for value in range(1, 6))
        rows.append(f'<label>{key}: <select data-score="{key}"><option value="">n/a</option>{options}</select></label>')
    return "\n".join(rows)


def build_html(result_dir: Path) -> str:
    summary = load_summary(result_dir)
    input_images = find_input_images(result_dir)
    input_img_html = "".join(
        f'<figure><img src="{html.escape(rel_link(path, result_dir))}" alt="input"><figcaption>{html.escape(path.name)}</figcaption></figure>'
        for path in input_images
    ) or "<p>No staged input image found.</p>"
    basic_info = html.escape(read_text(result_dir / "basic_info.txt"))
    summary_json = html.escape(json.dumps(summary, indent=2))
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in summary.get("warnings", [])) or "<li>None</li>"
    errors = "".join(f"<li>{html.escape(item)}</li>" for item in summary.get("errors", [])) or "<li>None</li>"
    verdict_options = "".join(f'<option value="{v}">{v}</option>' for v in VERDICTS)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PhysX Result Review - {html.escape(result_dir.name)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow: auto; max-height: 360px; }}
    label {{ display: block; margin: 8px 0; }}
    .missing, .errors {{ color: #b00020; }}
    .warnings {{ color: #8a5a00; }}
    textarea {{ width: 100%; min-height: 120px; }}
    button {{ padding: 8px 12px; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>PhysX Result Review: {html.escape(result_dir.name)}</h1>
  <p>Status: <strong>{html.escape(str(summary.get('status')))}</strong></p>

  <div class="grid">
    <section>
      <h2>Input image</h2>
      {input_img_html}
      <h2>Artifacts</h2>
      <ul>{artifact_list(result_dir)}</ul>
    </section>
    <section>
      <h2>Auto warnings</h2>
      <ul class="warnings">{warnings}</ul>
      <h2>Auto errors</h2>
      <ul class="errors">{errors}</ul>
    </section>
  </div>

  <section>
    <h2>VLM basic_info.txt</h2>
    <pre>{basic_info}</pre>
  </section>

  <section>
    <h2>Inspection summary</h2>
    <pre>{summary_json}</pre>
  </section>

  <section>
    <h2>Human checklist</h2>
    {checklist_html()}
  </section>

  <section>
    <h2>Scores</h2>
    {scores_html()}
  </section>

  <section>
    <h2>Verdict and notes</h2>
    <label>Verdict: <select id="verdict"><option value="">Select...</option>{verdict_options}</select></label>
    <textarea id="notes" placeholder="Notes for manual edit / rerun / accepted result..."></textarea>
    <br>
    <button onclick="downloadReview()">Download review_result.json</button>
  </section>

<script>
function collectReview() {{
  const checklist = {{}};
  document.querySelectorAll('[data-key]').forEach(el => checklist[el.dataset.key] = el.checked);
  const scores = {{}};
  document.querySelectorAll('[data-score]').forEach(el => scores[el.dataset.score] = el.value ? Number(el.value) : null);
  return {{
    result_dir: {json.dumps(str(result_dir))},
    inspection_status: {json.dumps(summary.get('status'))},
    checklist,
    scores,
    verdict: document.getElementById('verdict').value,
    notes: document.getElementById('notes').value,
    reviewed_at: new Date().toISOString()
  }};
}}
function downloadReview() {{
  const blob = new Blob([JSON.stringify(collectReview(), null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'review_result.json';
  a.click();
  URL.revokeObjectURL(a.href);
}}
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local HTML review page for a PhysX result directory.")
    parser.add_argument("result_dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")
    out = result_dir / "review.html"
    out.write_text(build_html(result_dir))
    print(out)


if __name__ == "__main__":
    main()
