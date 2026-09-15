# Egocentric Spatial Analysis

English | [简体中文](README.zh-CN.md)

A reusable pipeline for egocentric video preprocessing, VGGT-based 3D reconstruction, and viewpoint-aware spatial analysis.

**Input:** one or more labeled first-person videos (or an existing reconstruction directory).
**Output:** VGGT-compatible frame sequences, camera / point-cloud arrays, and a JSON report of relative geometric and visibility metrics.

[VGGT](https://github.com/facebookresearch/vggt) is an **external** reconstruction dependency. This repository is not a new 3D model, not a VGGT fork, and not a metric-scale reconstruction framework.

User-authored work in this repository is the engineering around VGGT: multi-video handling, preprocessing, reconstruction orchestration, serialization, and viewpoint-aware analysis on the resulting artifacts.

```text
egocentric video
        ↓
configurable video preprocessing
        ↓
VGGT-compatible frame preparation
        ↓
external VGGT reconstruction
        ↓
camera / point-cloud outputs
        ↓
viewpoint-aware spatial analysis
        ↓
relative geometric and visibility metrics
```

## What you can do with this code

- Handle multiple labeled videos in one run (`LABEL=path`; identity is never inferred from even/odd frame order)
- Probe container metadata (resolution, FPS, duration)
- Align frame rates with ffmpeg when streams differ
- Sample frames in a time window (`uniform`, `quality_weighted`, `complementary`)
- Score candidate frames (sharpness, exposure, contrast, optional ORB; NumPy fallback if OpenCV feature detectors are missing)
- Resize / crop / pad to VGGT-compatible square inputs (default 518)
- Assemble labeled frame folders into one reconstruction image directory
- Orchestrate VGGT inference with device selection and a memory cap (`--max-images`)
- Serialize cameras, depth, and points (`extrinsic.npy`, `intrinsic.npy`, `points_3d.npy`, optional PLY)
- Analyze camera geometry (OpenCV `[R|t]`, viewing angles)
- Compare viewpoints (elevation / yaw-style angles)
- Report relative height versus an up-axis percentile “ground”
- Run simplified frustum-style visibility
- Approximate ray proximity with a KD-tree (not mesh ray tracing)

Distances are **reconstruction units** unless you pass an external `--scale`. Metric scale is not assumed.

## Repository structure

```text
src/egocentric_spatial_analysis/
  preprocessing/     probe, fps align, sampling, quality, resize
  reconstruction/    VGGT wrapper (external package + user-supplied weights)
  analysis/          cameras, relative height, visibility, ray hits
scripts/             command-line entry points
docs/                methodology, collection principles, limitations
tests/               unit tests on synthetic arrays and images
examples/            input-format notes only; no private footage
```

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
pip install -e .
```

Video preprocessing needs a **working** OpenCV 4.x build that provides `cv2.VideoCapture` / `cv2.VideoWriter`. Declare **one** wheel:

- `opencv-python>=4.8,<5` (default in `requirements.txt`), or
- `opencv-python-headless` in the same 4.x range

Do not install both, and do not mix in `opencv-contrib-python`. OpenCV 5.x pulls `numpy>=2`, which conflicts with this project’s `numpy>=1.24,<2`. A leftover empty `cv2/` folder or a broken contrib install can shadow the real package: `import cv2` succeeds but video I/O attributes are missing, and video tests skip. Quality metrics still run via a NumPy fallback; frame extraction does not.

VGGT reconstruction is **not** installed by `pip install -r requirements.txt` or `pip install -e .`. Those commands do not download `model.pt`. Install the official VGGT package from upstream and obtain a licensed checkpoint yourself. This repository does not bundle VGGT source or weights.

```bash
# Optional torch / trimesh extras for this wrapper. Still not VGGT itself.
pip install -e ".[reconstruction]"

# VGGT (external). Follow upstream; a typical local install is:
#   git clone https://github.com/facebookresearch/vggt
#   pip install -e path/to/vggt
# Then download model.pt (or another licensed checkpoint) from Meta / Hugging Face
# and pass --weight-path. --from-pretrained may download facebook/VGGT-1B; that is
# an explicit opt-in, not part of pip install -r requirements.txt.
```

Match torch / CUDA to your machine. Reconstruction extras do not replace a VGGT install.

## Basic usage

Preprocess labeled observer videos:

```bash
python scripts/preprocess.py extract \
  --video observer_a=path/to/a.mp4 \
  --video observer_b=path/to/b.mp4 \
  --start-time 10 --end-time 20 \
  --num-frames 10 \
  --strategy quality_weighted \
  --output-dir outputs/frames
```

Resize and assemble a reconstruction folder:

```bash
python scripts/preprocess.py resize --input-dir outputs/frames/observer_a --output-dir outputs/images --size 518
python scripts/preprocess.py assemble \
  --frames observer_a=outputs/frames/observer_a \
  --frames observer_b=outputs/frames/observer_b \
  --output-dir outputs/images
```

Reconstruct (local VGGT install and weights required):

```bash
python scripts/reconstruct.py \
  --image-dir outputs/images \
  --output-dir outputs/reconstruction \
  --weight-path path/to/VGGT/model.pt \
  --device auto \
  --max-images 20
```

Analyze with explicit labels, or filenames of the form `frame_000_<label>_...`:

```bash
python scripts/analyze.py \
  --reconstruction-dir outputs/reconstruction \
  --output-dir outputs/analysis \
  --labels observer_a observer_b observer_a observer_b
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Geometry, sampling, and analysis tests use synthetic arrays. The video extraction test writes a tiny synthetic clip with OpenCV `VideoWriter` and reads it with `VideoCapture` (no private footage, no mocked capture). With a working OpenCV 4.x wheel it should run, not skip. If `cv2` is missing those attributes, that test **skips** instead of faking a pass.

## VGGT dependency and attribution

Reconstruction uses **VGGT: Visual Geometry Grounded Transformer** (Wang et al., CVPR 2025). Install and cite the official project. Code, demonstration materials, and checkpoints remain under the VGGT license and acceptable-use policy. Users must obtain checkpoints from Meta / Hugging Face and comply with those terms. This project does not grant rights to VGGT source or weights.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Original code in this repository is classified as an independent VGGT wrapper (`INDEPENDENT_WRAPPER`) and is technically ready for its own top-level license (`READY_FOR_INDEPENDENT_TOP_LEVEL_LICENSE`). No SPDX `LICENSE` file is included yet; choose MIT / Apache-2.0 / BSD (or another) separately. Do not treat the VGGT license as covering the wrapper, preprocessing, or analysis layers. Keep [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data privacy

Original egocentric footage, identifiable people, and participant-related files are **not** included. Do not commit raw videos, unredacted frames, or large point clouds. Keep private data outside the repository (for example `data/private/`, which is gitignored).

## Limitations

- VGGT is external; reconstruction quality and licensing are upstream concerns.
- Metric scale is not assumed. Default analysis is relative (angles, ratios, visibility fractions).
- “Ground” is an up-axis percentile, not a fitted plane.
- Visibility is a simplified angular frustum; ray hits are KD-tree proximity, not a renderer.
- This is an engineering / research pipeline, not a validated population-level study.
- Private study footage is excluded, so unpublished numbers from that footage cannot be reproduced here.

Details: [docs/limitations.md](docs/limitations.md).

## Origins / exploratory use case

The pipeline was first assembled around a private dual wearable-camera capture, including an exploratory adult/child street-scene comparison. That session is **historical context only**. It is not a completed validation study, not a claim about children’s spatial perception, and not a constraint on the API (observers are always caller-supplied labels).
