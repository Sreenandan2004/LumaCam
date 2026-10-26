# LumaCam Change Log

All notable changes to this project are recorded here. **Logging is mandatory** (see
[AGENTS.md § 0, Master Rules](AGENTS.md)). A change without an entry is not complete.

Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each entry records
**What / Why**, **Files**, **Verification**, and **Impact / Follow-ups**. Proposals that need another
owner's sign-off go under **Open Decisions**, not into code.

---

## [Unreleased]

### Added — Evaluation benchmarking suite & automated report generator (Nandan)

- **What / Why:** Completed Step 2 of Thursday presentation plan: enhanced model/evaluate.py with multi-stage evaluation (comparing Linear ISO Gain, LumaCam Generator, and LumaCam + Color Science), persistent JSON/Markdown report generation in 
esults/, UTF-8 Windows terminal support, and added unit test suite 	ests/unit/test_evaluate.py.
- **Files:** model/evaluate.py, 	ests/unit/test_evaluate.py, CHANGELOG.md.
- **Verification:** Ran python model/evaluate.py on GTX 1650 (70.4 FPS, 14.20 ms latency). All 49 unit tests pass (pytest tests/unit -q).
- **Impact / Follow-ups:** Produces publication and slide-ready quantitative benchmark tables for Thursday reviews.

### Added — Interactive Streamlit demo dashboard and evaluation suite (Nandan)

- **What / Why:** Needed a real-time, interactive demonstrator for Thursday capstone reviews showing live exposure slider (α) modulation, 4-way computational photography comparison (Raw Bayer -> Linear Gain -> LumaCam GAN -> Color Science), explainable AI Attention Gate heatmaps, and latency/PSNR metrics cards.
- **Files:** pp.py, model/evaluate.py, 
equirements.txt, AGENTS.md, Progress.md, CHANGELOG.md.
- **Verification:** Tested on NVIDIA GeForce GTX 1650 (CUDA) with checkpoints/lumacam_best.pt. All 46 unit tests pass (pytest tests/unit -v).
- **Impact / Follow-ups:** Provides the team with an interactive GUI ready to present to faculty; unblocks visual evaluation while mobile integration proceeds.

### Branch Sequence — Lakshmi's review fixes (2026-10-12)

Five stacked branches, each based on the previous one. Merge them into `dev` **in this order**:

| # | Branch | Scope |
|---|---|---|
| 1 | `feature/docs-engineering-rules` | Master Rules in AGENTS.md, this change log |
| 2 | `feature/dataset-sid` | SID loader fixes: list parsing, `data_root`, official split, input clipping |
| 3 | `feature/opt-onnx-export` | ONNX exporter (`model/export_onnx.py`), Windows-installable requirements |
| 4 | `feature/model-color-science` | Post-inference color science (`model/postprocess/`) |
| 5 | `feature/docs-api-spec` | I/O contract (`docs/api_specifications.md`), Progress/plan doc fixes |

### Added — Master Rules and mandatory change log (branch 1, Lakshmi)

- **What / Why:** The project had no single place recording *what* changed and *why*. Progress.md
  tracks milestones, not decisions or behavior changes, and the handoff docs had already drifted
  from the code (e.g. `lakshmi.md` still assigned the SID loader that was already merged).
  Added `AGENTS.md § 0 Master Rules`:
  - every change is logged here;
  - affected docs are updated in the same branch;
  - one logical change per branch, with stacked branch sequences;
  - tests with every behavior change, fail-fast errors, a single source of truth for constants;
  - no unilateral changes to cross-team contracts.

  Also added the `feature/docs-<task>` branch prefix and the branch-sequence workflow (`AGENTS.md § 6.2`).
- **Files:** `AGENTS.md`, `CHANGELOG.md` (new).
- **Verification:** Documentation only.
- **Impact / Follow-ups:** All members and AI assistants must follow § 0 from now on.

### Fixed — SID dataset loader correctness (branch 2, Lakshmi; owner of original code: Nandan)

1. **Crash on official list files.**
   - *Problem:* Official SID lists contain lines like
     `./Sony/short/00001_00_0.1s.ARW ./Sony/long/00001_00_10s.ARW ISO200 F8`. The loader treated
     columns 3–4 as exposure times, so `float("ISO200")` raised `ValueError`.
   - *Fix:* Exposure is now always parsed from the filenames.
2. **Two meanings of `data_root`.**
   - *Problem:* Directory-scan mode expected the `Sony/` folder itself. List mode expected its parent, and with the
     default config it resolved to `data/Sony/Sony/short/...`.
   - *Fix:* `data_root` now always means the `Sony/` folder. List entries keep only `<short|long>/<file>`
     (`_resolve_list_path`), and the old `.lstrip("./")` (which strips characters, not a prefix) is gone.
3. **Train/val/test leakage.**
   - *Problem:* Directory scanning, which the Kaggle notebook uses, mixed all scenes. Test scenes (`1xxxx`) were
     trained on, and validation was the first 20 *training* files, so reported validation PSNR was inflated.
   - *Fix:* Added `split=` (`train`/`val`/`test`), which filters by SID scene-ID prefix (`SID_SPLIT_PREFIXES`:
     `0` train, `1` test, `2` val). `model/train.py` and `notebooks/train_lumacam_kaggle.ipynb` now pass `split="train"`
     and `split="val"`. Both list files in `train_config.yaml` default to `null`, so training scans the
     directory with the official split.
4. **Unclipped amplified input.**
   - *Problem:* Chen et al. (SID) feed `min(packed * ratio, 1.0)`. The loader fed `packed * ratio`, so bright pixels reached
     ~300. That is out of range for a network whose targets are in [0, 1], and it would ruin the INT8
     calibration range.
   - *Fix:* Added `AMPLIFIED_INPUT_CEILING = 1.0` in `raw_processing.py` as the single source of truth.
     It is used by `SIDSonyDataset`, by `DummySIDDataset` (was clipped at 10.0), and by
     `apply_exposure_amplification`, which now clamps as its docstring always claimed.
5. **`apply_exposure_amplification` broadcast bug.**
   - *Problem:* A `(B, 1)` ratio broadcast against the trailing `(H, W)` axes instead of the batch axis.
   - *Fix:* It is now reshaped to `(B, 1, 1, 1)`.
6. **Silent fallbacks removed (fail fast).**
   - Unparseable exposure names raise `ValueError` instead of returning `1.0`.
   - A missing `list_file` raises `FileNotFoundError` instead of silently falling back to scanning.
   - An empty dataset raises `ValueError` with a hint about `data_root`.

- **Files:** `model/datasets/sid_dataset.py`, `model/datasets/raw_processing.py`,
  `model/datasets/__init__.py`, `model/train.py`, `model/configs/train_config.yaml`,
  `notebooks/train_lumacam_kaggle.ipynb` (dataset cell + checkpoint `config` key, 7 lines),
  `tests/unit/test_sid_dataset.py` (new), `Progress.md`.
- **Verification** (Windows 10, Python 3.12, torch 2.14.1+cpu):
  - `python -m unittest tests.unit.test_model_forward tests.unit.test_sid_dataset` → **22/22 OK**
    (10 existing + 12 new).
  - The new tests catch real bugs: run against the pre-fix `origin/dev` code, **11 of 12 fail**, including
    `could not convert string to float: 'ISO200'`, the `(B, 1)` broadcast `RuntimeError`, and a dummy
    input max of `2.44 > 1.0`.
  - `python model/train.py --dry-run --epochs 1 --batch-size 1 --device cpu` → completes
    (val PSNR 11.79 dB untrained, 437 s/epoch on CPU).
- **Impact / Follow-ups:**
  - **Behavior change:** inputs are now clipped to 1.0. Checkpoints trained before this change saw
    unclipped inputs, so **retrain (or fine-tune) before final benchmarking**. Validation PSNR from earlier
    notebook runs was measured on training scenes and should not be reported.
  - The notebook's best checkpoint now also stores `config.model`, matching `model/train.py`, so the
    ONNX exporter can rebuild the generator.
  - Nandan (owner of `sid_dataset.py` / `train.py`) to review.

### Added — ONNX exporter and Windows-installable requirements (branch 3, Lakshmi)

- **What / Why:** Task 6. INT8 quantization (Task 7) could not start without an ONNX graph.
  `model/export_onnx.py` adds:
  - **`GeneratorExportWrapper`:** an explicit `input_bayer` + `alpha_exposure` → `enhanced_rgb` contract.
    *Correction to the 2026-10-12 review:* PyTorch 2.14 does export the raw `(out, None)` generator,
    because the tracer drops the `None`. So the wrapper makes the contract explicit; it does not fix a crash.
  - **A fixed input shape by default** (`--dynamic` optional). The attention gates' `F.interpolate` resizes
    become constants, which suits LiteRT and INT8.
  - **`validate_input_size`:** packed H and W must be multiples of 16, because of the 4 MaxPool stages.
  - **`load_generator`:** accepts `train.py` and notebook checkpoints and bare state dicts, rebuilds the model
    from `config.model`, loads with `strict=True`, and requires an explicit `--random-init` for untrained weights.
  - **`dynamo=False` pinned:** `torch.onnx.export` now defaults to `dynamo=True` (verified on 2.14.1).
  - **Checks:** `onnx.checker` and a parity check against PyTorch on realistic inputs (values in [0, 1], α = 100).
- **Also fixed:** `requirements.txt` installed `ai-edge-litert` on Windows with Python < 3.13, but it ships
  Linux/macOS wheels only, so `pip install -r requirements.txt` failed on Windows. It is now skipped on
  Windows (run LiteRT conversion on Kaggle/Colab/WSL). Added `pytest`.
- **Environment note (not a repo change):** on this laptop, PyTorch (`c10.dll`) and onnxruntime failed with
  `WinError 1114` because the Microsoft VC++ runtime was 14.26 (2020). Installing the current
  VC++ 2015–2022 Redistributable x64 (14.51) fixed both. Anyone on Windows with this error should do the same.
- **Files:** `model/export_onnx.py` (new), `tests/unit/test_export_onnx.py` (new), `requirements.txt`,
  `Progress.md`.
- **Verification:**
  - `python -m pytest tests/unit/test_export_onnx.py -v` → **8 passed**: fixed and dynamic export, parity
    at another size and batch (32×96, B=2), invalid sizes, and checkpoint formats.
  - `python model/export_onnx.py --checkpoint checkpoints/lumacam_best.pt --test` on a `train.py` dry-run
    checkpoint → 31.15 MB, opset 17, `onnx.checker` passed, **max |diff| = 8.9e-08** (PASS).
  - Missing checkpoint and `--height 100` → clear error, exit code 1.
- **Impact / Follow-ups:**
  - `edge_opt/models/*.onnx` stays untracked (`.gitignore`; see OD-4 in `docs/api_specifications.md`).
  - The `TracerWarning` from `attention_gate.py:57` is benign (explained in the module docstring).
  - Whoever owns Task 7 (OD-3) can now start from `edge_opt/models/lumacam_fp32.onnx`.

### Added — Post-inference color science (branch 4, Lakshmi)

- **What / Why:** Task 13. `lakshmi.md` specified `I^(1/2.2)` gamma on the model output. But training targets
  come from `rawpy.postprocess(use_camera_wb=True, no_auto_bright=True)`. The effective default gamma, verified
  on rawpy 0.27.1, is `(1/2.222, 4.5)`: the BT.709 curve, in the sRGB color space. So the model already
  outputs gamma-encoded, white-balanced sRGB, and a second gamma would double-encode it (mid-gray
  0.50 → 0.73, washed out). `model/postprocess/color_science.py` therefore provides:
  - `srgb_to_linear` / `linear_to_srgb`: IEC 61966-2-1 transfer functions. Gamma encoding happens only
    for inputs declared linear.
  - `gray_world_awb`: Gray-World white balance in **linear** light, where channel gains are physically
    meaningful. Blend strength, gains capped to [1/max_gain, max_gain], clipped pixels excluded,
    and a black-image guard.
  - `adjust_saturation`: an exact closed form for HSV saturation scaling, `c' = V − k(V − c)`, with
    `k ≤ V/(V − min)` per pixel. Hue and value are preserved and colors stay in gamut, without an HSV round trip.
  - `apply_color_pipeline` (torch, any `(..., 3, H, W)`) and `apply_color_pipeline_np` (`(H, W, 3)`).
    Defaults: AWB strength 0.5 (the model already learned camera white balance) and saturation ×1.15.
    There is one implementation in PyTorch, and the NumPy path wraps it.
- **Files:** `model/postprocess/__init__.py`, `model/postprocess/color_science.py`,
  `tests/unit/test_color_science.py` (all new), `AGENTS.md` (§ 4 Workstream 3 gamma note, § 5 tree), `Progress.md`.
- **Verification:** `python -m pytest tests/unit/test_color_science.py -v` → **16 passed**. The tests cover:
  - sRGB round trip and known values;
  - a gray image unchanged by AWB, and a color cast removed;
  - gain cap, black-image safety, and saturation identity, value and hue preservation;
  - neutral-settings identity, which proves there is no double gamma;
  - input validation.
- **Impact / Follow-ups:**
  - AGENTS.md § 4 (Workstream 3) now says gamma encoding applies only to linear data. The generator's output must
    not be gamma-corrected again.
  - Veni/Nandan: the Android port of this pipeline (Kotlin or C++) should follow `docs/api_specifications.md` § 6.

### Docs — API contract, open decisions, and stale planning docs (branch 5, Lakshmi)

- **What / Why:** The tensor contract between dataset, model, ONNX artifact and Android app existed only
  implicitly in code, and several planning docs were stale or wrong.
  - **`docs/api_specifications.md` (new):** the single source of truth for:
    - `input_bayer` (shape, `[R, G1, B, G2]` channel order, normalization, 1.0 ceiling, multiple-of-16 rule,
      even-offset crops, reading sensor constants from Camera2 instead of hard-coding Sony's);
    - `alpha_exposure` and `enhanced_rgb` (gamma-encoded sRGB, camera WB baked in);
    - the checkpoint format, the ONNX artifact, and the post-processing defaults.
  - **Open Decisions** (in the spec, § 7), recorded instead of changed in code (Master Rule 5):
    - **OD-1:** the app's α mapping (0.5–3.0) vs the trained α (100–300).
    - **OD-2:** α is applied twice (input amplification + FiLM).
    - **OD-3:** who owns INT8 (Task 7) and benchmarks (Task 8). `TEAM_PLAN.md`/`me.md` and `Progress.md` disagree.
    - **OD-4:** `.onnx`/`.tflite` are gitignored, but AGENTS.md says to track them with Git LFS.
    - **OD-5:** 8-bit vs 16-bit ground truth.
  - **`TEAM_PLAN.md`:** the handoff links pointed to `file:///C:/Users/imsre/.gemini/...` on another machine.
    They are now relative links, plus pointers to the change log and the API spec.
  - **`lakshmi.md`:**
    - status banner: Task 2 was done by Nandan and fixed here, the loader returns a tuple not a dict,
      Tasks 6 and 13 are done, and the spec's extra gamma is superseded;
    - checklist corrected (`d[0][0]`, not `d[0]['input']`) and ticked.
  - **`veni.md`:** OD-1 warning on the EV→α mapping, so the slider is not wired to an out-of-distribution α.
  - **`Progress.md`:**
    - header (date, active branch);
    - task graph (Tasks 6 and 13 done; Tasks 7 and 8 owner TBD);
    - next steps (merge order, retrain, open decisions);
    - test count 46/46.
- **Files:** `docs/api_specifications.md` (new), `TEAM_PLAN.md`, `lakshmi.md`, `veni.md`, `Progress.md`, `CHANGELOG.md`.
- **Verification:**
  - Documentation only.
  - The facts it cites were checked: `python -m pytest tests/unit -q` → **46 passed**.
  - `DummySIDDataset()[0][0].shape` → `torch.Size([4, 256, 256])`.
  - rawpy 0.27.1 default gamma is `(0.45, 4.5)`.
  - `torch.onnx.export` defaults to `dynamo=True` on torch 2.14.1.
- **Impact / Follow-ups:** Nandan and Veni to decide OD-1 to OD-5. Until OD-1 is settled, the app must not
  map the EV slider to α.

### Open Decisions (summary — details in `docs/api_specifications.md` § 7)

| ID | Question | Owners | Status |
|---|---|---|---|
| OD-1 | α units/range from the app vs trained range 100–300 | Nandan, Veni | Half resolved: model normalizes α (`f56bf4c`); app range still open, blocks slider→α wiring |
| OD-2 | α applied to both the input and FiLM | Nandan | Open |
| OD-3 | Owner of INT8 (Task 7) and edge benchmarks (Task 8) | Nandan | ✅ Resolved: Lakshmi (`report.md` § 5) |
| OD-4 | Git LFS for `.onnx`/`.tflite` vs `.gitignore` | Nandan | Open |
| OD-5 | 8-bit vs 16-bit ground truth | Nandan | Open (next retrain) |
| OD-6 | ONNX input name `alpha_exposure` vs `alpha_cond`; fixed vs dynamic shapes in `report.md` | Nandan, Lakshmi | Open |

### Docs — OD-3 resolved, OD-6 added after rebasing onto `report.md` (branch 5, Lakshmi)

- **What / Why:** Nandan pushed `history.md` and `report.md` to `dev` (`cb97788`..`05e5a48`). `report.md` § 5 makes
  Lakshmi the *Edge Optimization & LiteRT Runtime Lead*. That resolves **OD-3**: INT8 PTQ, the calibration set,
  the LiteRT Android delegates and Pixel 7 profiling are hers, and color science moves to Veni. The report also
  names the second ONNX input `alpha_cond` and calls the spatial dimensions dynamic. Both conflict with the
  exporter (`alpha_exposure`, fixed shape by default), so this is recorded as the new **OD-6** instead of changing
  code or Nandan's report.
- **Files:** `docs/api_specifications.md` (§ 7), `CHANGELOG.md`.
- **Verification:** Documentation only. All 5 branches were rebased onto `origin/dev` `05e5a48` (one `Progress.md`
  conflict, resolved by keeping both sides' rows). `python -m pytest tests/unit -q` → 46 passed at the stack tip.
- **Impact / Follow-ups:** Nandan and Lakshmi to settle OD-6 before the Android LiteRT wrapper is written.

### Docs — OD-1 half resolved by `f56bf4c`, old checkpoints incompatible (branch 5, Lakshmi, 2026-10-22)

- **What / Why:** Nandan's `f56bf4c` on `dev` (GAN stabilization) changes two things the contract describes:
  - FiLM now feeds its MLP `log2(max(α, 1)) / 10` instead of the raw ratio. That is the model-side half of OD-1's
    proposal. The app side stays open: α still multiplies the input, and the clamp maps the app's planned
    α ∈ [0.5, 3.0] to 0–0.16, far from the trained 0.66–0.82. The model input is still the raw ratio.
  - Every Attention Gate gains a GroupNorm (`norm.weight` / `norm.bias`), so checkpoints from before it fail a
    strict load.
- **Files:** `docs/api_specifications.md` (§ 2.2, § 4, § 7 OD-1), `CHANGELOG.md`.
- **Verification:** Documentation only. Read `f56bf4c`'s diff of `model/networks/conditioning.py` and
  `attention_gate.py`. The values follow from it: log2(100)/10 = 0.664, log2(300)/10 = 0.823, log2(3)/10 = 0.158.
  - Merging the 7 stacked branches into `origin/dev` `f56bf4c` in PR order (temporary worktree) gives no conflicts.
    `pytest tests/unit` passes after every merge: 10 → 10 → 22 → 30 → 46 → 46 → 62.
  - `load_generator("checkpoints/lumacam_best.pt")` (2026-10-12) on the merged code raises
    `RuntimeError: Missing key(s) ... "ag4.norm.weight", "ag4.norm.bias", ...`.
- **Impact / Follow-ups:**
  - **Veni + Nandan:** agree on the on-device α computation (OD-1 proposal) before wiring the slider.
  - **Nandan:** retrain on Kaggle; `checkpoints/lumacam_best.pt` from 2026-10-12 no longer loads.
