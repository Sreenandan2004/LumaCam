# LumaCam Change Log

All notable changes to this project are recorded here. **Logging is mandatory** (see
[AGENTS.md § 0, Master Rules](AGENTS.md)). A change without an entry is not complete.

Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each entry records
**What / Why**, **Files**, **Verification**, and **Impact / Follow-ups**. Proposals that need another
owner's sign-off go under **Open Decisions**, not into code.

---

## [Unreleased]

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
