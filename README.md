# lerobot

Custom `lerobot` workspace for local robot control, teleoperation, and hardware experiments.

This repo is based on the upstream Hugging Face `lerobot` project, with additional local code under:

- `examples/`
- `src/lerobot/model/SO101Robot.py`
- `src/lerobot/robots/xlerobot/`

## Clone

```bash
git clone https://github.com/abhinav24jha/lerobot
cd lerobot
```

## Linux Setup

This repo is being transferred to a non-Mac laptop, so use the Linux dependency file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements-ubuntu.txt
```

If package builds fail on Linux, install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ffmpeg \
  git \
  libavcodec-dev \
  libavdevice-dev \
  libavfilter-dev \
  libavformat-dev \
  libavutil-dev \
  libswresample-dev \
  libswscale-dev \
  pkg-config \
  python3-dev \
  python3-venv
```

## Notes

- `.venv`, `outputs/`, `data/`, and caches are intentionally not tracked.
- The original upstream remote is kept locally as `upstream`.
- Hardware access on Linux may require USB permissions depending on the motor controller and serial device.

## Upstream

Original project: https://github.com/huggingface/lerobot
