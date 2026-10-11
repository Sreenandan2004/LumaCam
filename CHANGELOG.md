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
