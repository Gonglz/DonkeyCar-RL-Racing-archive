> [!NOTE]
> This repository is an archived snapshot of the original development history. The polished portfolio version now lives at https://github.com/Gonglz/DonkeyCar-RL-Racing.
# mysim

This directory is set up to work well as the repo root for simulator-side
development across multiple machines.

Tracked in git:
- `manage.py`
- `train.py`
- `calibrate.py`
- `config.py`
- `myconfig.py`
- any future code / scripts you add here

Ignored from git:
- `data/`
- `models/`
- `logs/`
- `unitylog.txt`
- `*.zip`
- editor/cache files

Recommended multi-machine workflow:
1. Clone the same repo on each machine.
2. Keep simulator code and shared config in this directory.
3. Keep generated tubs, trained models, logs, and local experiments out of git.
4. If a machine needs local-only config, put it in `myconfig.local.py` instead
   of changing tracked files first.

Notes:
- This repo layout assumes `donkeycar` is available in the Python environment.
- If you later want to sync models or datasets, use separate storage instead of
  checking large binaries into git by default.
- V16 dual-domain obstacle curriculum details live in `docs/V16_CURRICULUM.md`.
