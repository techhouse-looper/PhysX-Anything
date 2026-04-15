# PhysX-Anything Evaluation Automation Plan

## Current state

The pipeline can now run end-to-end from a single image:

```text
image -> 1_vlm_demo.py -> 2_decoder.py -> 3_split.py -> 4_simready_gen.py
```

Validated local examples:

- `sissors.png` -> generated `sample.glb`, split OBJ parts, `basic.urdf`, `basic.xml`.
- `microwave.png` -> generated `sample.glb`, split OBJ parts, `basic.urdf`, `basic.xml`.

Known quality gap:

- Some meshes are visually wrong even when the pipeline succeeds.
- Example: one scissor handle can preserve a loop while the other handle is filled/solid.
- Single-image ambiguity also causes incomplete or inaccurate back-side geometry.

The next work should focus on **result evaluation**, not more generation knobs.

---

## Recommended PR order

### PR #4 — Core inspection report

**Why first:** every later review or evaluator needs a stable machine-readable summary of what was generated.

Add:

```text
scripts/inspect_pipeline_result.py
```

Input:

```bash
python scripts/inspect_pipeline_result.py <result_dir>
```

Example:

```bash
python scripts/inspect_pipeline_result.py \
  pipeline_runs/microwave_png_<hash>/test_demo/microwave_png_<hash>
```

Scope:

- Required file checks:
  - `basic_info.txt`
  - `basic_info.json`
  - `sample.glb`
  - `basic.urdf`
  - `basic.xml`
  - `objs/`
- File size sanity checks.
- `basic_info.json` parse check.
- `coord_*.txt` malformed/range token check.
- `ind_*.npy` and `allind.npy` stats:
  - shape
  - coordinate min/max
  - voxel count
  - out-of-range coordinates
- OBJ part stats:
  - vertex count
  - face count
  - file size
- GLB load check via `trimesh`.
- MuJoCo XML load check if `mujoco` is installed; otherwise mark as skipped.
- Save:

```text
inspection_summary.json
inspection_report.md
```

Acceptance criteria:

- Running on both existing local `sissors` and `microwave` result dirs produces JSON + Markdown reports.
- Missing required files cause `FAIL` status.
- Warnings do not fail the script unless there are hard errors.
- No raw CSV telemetry is committed.

Verification commands:

```bash
conda run -n physx-decoder python -m py_compile scripts/inspect_pipeline_result.py
conda run -n physx-decoder python scripts/inspect_pipeline_result.py <result_dir>
```

---

### PR #5 — Part topology and semantic consistency checks

**Why second:** this directly targets the scissor handle problem, but should build on PR #4's report schema.

Extend `inspect_pipeline_result.py` with part-quality heuristics.

Add checks:

1. **Projected hole detection**
   - Load each `objs/<id>/<id>.obj`.
   - Project mesh vertices/faces onto XY/XZ/YZ planes.
   - Rasterize to a binary mask.
   - Count holes using filled-mask minus original-mask components.

2. **Handle/opening expectation**
   - If part name contains keywords such as:

```text
handle, grip, loop, frame
```

   then at least one projected view should show a meaningful hole/opening.

3. **Repeated semantic part consistency**
   - If two or more parts share a normalized name, compare:
     - voxel count
     - OBJ vertex/face count
     - bounding-box extents
     - projected hole count

Target warning for scissors:

```text
Repeated part "Blade Handle Set" has inconsistent projected hole counts: [1, 0]
Possible filled handle loop in part <id>.
```

Acceptance criteria:

- Scissor result surfaces a warning if one handle part has no projected hole while its counterpart does.
- Microwave result does not fail just because some parts are solid panels.
- The report distinguishes `errors` from `warnings`.

---

### PR #6 — Human review page

**Why third:** once auto-inspection produces a stable JSON report, a browser-based review form can present it cleanly.

Add:

```text
scripts/build_review_page.py
```

Input:

```bash
python scripts/build_review_page.py <result_dir>
```

Output:

```text
review.html
```

HTML page should include:

- Input image path or staged image preview.
- Links to generated artifacts:
  - `sample.glb`
  - `basic.urdf`
  - `basic.xml`
  - `inspection_summary.json`
- Auto inspection summary.
- Human checklist:
  - object identity
  - global geometry
  - missing back/side geometry
  - holes/openings preserved
  - major parts present
  - part split reasonable
  - joint type plausible
  - joint axis plausible
  - MuJoCo load status
- Scores from 1–5.
- Freeform notes.
- JS button to download `review_result.json`.

Acceptance criteria:

- `review.html` opens locally without a server.
- Human can fill checklist and download review JSON.
- Page works with both scissors and microwave result dirs.

---

### PR #7 — Render previews

**Why fourth:** the human review page is more useful if it can display generated previews.

Add:

```text
scripts/render_result_preview.py
```

Output:

```text
renders/front.png
renders/side.png
renders/top.png
renders/iso.png
```

Possible implementation paths:

- Start with `trimesh`/`open3d` best-effort rendering.
- If headless rendering is unreliable, gracefully skip and report the reason.
- Do not block inspection if rendering dependencies are unavailable.

Acceptance criteria:

- Produces at least one preview image for local scissors or microwave output, or records a clear skipped status.
- `build_review_page.py` displays previews when available.

---

### PR #8 — VLM evaluator

**Why later:** VLM evaluation is useful but introduces model/API dependencies and should not block local deterministic inspection.

Add optional evaluator:

```text
scripts/evaluate_with_vlm.py
```

Inputs:

- original/staged image
- generated render previews
- `inspection_summary.json`
- `basic_info.txt`

Output:

```text
vlm_eval.json
```

Evaluation rubric:

- object identity
- global shape
- part completeness
- holes/openings
- articulation plausibility
- material plausibility
- simulation readiness

Important rule:

```text
VLM evaluator is advisory only. Human review remains the final verdict.
```

Acceptance criteria:

- Optional dependency path is documented.
- If evaluator credentials/model are unavailable, script exits with a clear skipped status.

---

## Human checklist baseline

Use this checklist in Markdown and HTML review forms.

### Object identity

- [ ] Generated object matches input category.
- [ ] Overall proportions are plausible.
- [ ] Object is not replaced by a generic shape prior.

### Global geometry

- [ ] Front view is plausible.
- [ ] Side/back geometry is not obviously missing or collapsed.
- [ ] Thin structures are not excessively thickened.
- [ ] Hollow structures are not accidentally filled.

### Parts

- [ ] Major semantic parts are present.
- [ ] Part count is reasonable.
- [ ] Repeated/symmetric parts have comparable quality.
- [ ] Small parts are not hallucinated into large structures.

### Holes/openings

- [ ] Scissor handles / loops are open where expected.
- [ ] Microwave door/window/frame openings are not incorrectly filled.
- [ ] Glass/window/panel surfaces are not implausibly thick.

### Articulation

- [ ] Joint type is plausible.
- [ ] Joint axis position is plausible.
- [ ] Joint direction is plausible.
- [ ] Joint range is plausible.

### Simulation readiness

- [ ] `basic.xml` loads in MuJoCo or load failure is explained.
- [ ] `basic.urdf` exists and is non-empty.
- [ ] Mesh complexity is not extreme for interactive simulation.
- [ ] Visual/collision grouping is plausible.

### Verdict

Choose one:

```text
PASS
PASS_WITH_WARNINGS
NEEDS_MANUAL_EDIT
FAIL_RERUN_WITH_BETTER_IMAGE
FAIL_PIPELINE
```

---

## Implementation principles

1. Keep generation and evaluation separate.
2. Prefer deterministic local checks before VLM-as-judge.
3. Do not commit runtime outputs from `pipeline_runs/`.
4. Reports should be reproducible from result directories.
5. Scripts should fail loudly on hard errors and warn on quality concerns.
6. Human review should be structured enough to compare experiments later.

---

## Immediate next PR recommendation

Start with **PR #4: Core inspection report**.

Reason:

- It has the least dependency risk.
- It creates the JSON schema needed by every later step.
- It gives immediate value for the already-generated scissors and microwave runs.

Suggested branch:

```bash
git switch -c result-inspection
```

First implementation target:

```text
scripts/inspect_pipeline_result.py
```
