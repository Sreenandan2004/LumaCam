import os

with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
    content = f.read()

target = '## [Unreleased]\n'
entry = """## [Unreleased]

### Added — Checkpoint resuming support in Kaggle notebook for multi-session team training (Team 21)

- **What / Why:** Upgraded `notebooks/train_lumacam_kaggle.ipynb` to support seamless checkpoint resuming across Kaggle GPU sessions and teammate accounts. Added `RESUME_CHECKPOINT` configuration cell that restores generator and discriminator weights, optimizer states, epoch counter, and best validation PSNR, allowing training to be split across multiple sessions or accounts (e.g. Sreenandan, Sreelakshmi, Sreeveni) to leverage the team's combined 90 free GPU hours/week without losing progress.
- **Files:** `notebooks/train_lumacam_kaggle.ipynb`, `scripts/update_kaggle_notebook_resume.py`, `CHANGELOG.md`.
- **Verification:** Verified notebook JSON structure and checkpoint load logic; all 56 unit tests pass (`pytest tests/unit -q`).
- **Impact / Follow-ups:** Unblocks distributed multi-account cloud training to scale model convergence beyond 100+ epochs.
"""

if target in content:
    content = content.replace(target, entry, 1)
    with open('CHANGELOG.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print('CHANGELOG updated successfully')
else:
    print('Target not found')
