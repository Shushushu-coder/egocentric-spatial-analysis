# Egocentric Spatial Analysis

An exploratory pipeline for reconstructing egocentric video and analyzing viewpoint-dependent visibility and spatial geometry using [VGGT](https://github.com/facebookresearch/vggt) as an external reconstruction backend.

This repository is **not** a new 3D foundation model. It packages:

- configurable video preprocessing
- a thin VGGT inference wrapper
- viewpoint-aware spatial analysis on reconstruction artifacts

Observer identity (for example two head-mounted cameras) is always supplied as an explicit label. The pipeline does not infer roles from even/odd frame order.

## What this repository does

1. Probe videos, optionally align frame rates, sample frames, and resize them for VGGT.
2. Call an installed VGGT package to estimate cameras, depth, and point maps.
3. Compare labeled viewpoints using angles, visibility ratios, relative height, and ray hits.

Distances are **reconstruction units** unless you pass an external scale. Metric scale recovery is not treated as a reliable built-in capability.

## Pipeline

```text
raw egocentric video
        ↓
configurable preprocessing
        ↓
VGGT-compatible frames
        ↓
external VGGT reconstruction
        ↓
reconstruction artifacts (.npy)
        ↓
viewpoint-aware spatial analysis
        ↓
relative visibility / FOV / geometric metrics
```

## Repository structure

```text
src/egocentric_spatial_analysis/
  preprocessing/     video probe, fps align, quality sampling, resize
  reconstruction/    VGGT wrapper (external dependency)
  analysis/          cameras, visibility, height, FOV
scripts/             command-line entry points
docs/                methodology, collection principles, limitations
tests/               unit tests on synthetic data
examples/            usage notes only; no private footage
```

## Installation

Python 3.10+ is required. Create an environment, then:

```bash
pip install -r requirements.txt
pip install -e .
```

VGGT reconstruction needs the official package and a licensed checkpoint. Install VGGT from the upstream project and obtain weights yourself. This repository does not bundle or re-host VGGT source or `model.pt`.

Optional reconstruction extras (torch versions should match your CUDA setup):

```bash
pip install -e ".[reconstruction]"
```

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

Reconstruct (requires a local VGGT install and weights):

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

Run tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## VGGT dependency and attribution

Reconstruction uses **VGGT: Visual Geometry Grounded Transformer** (Wang et al., CVPR 2025). Install and cite the official project. Weights are subject to the VGGT license and acceptable-use policy. Users must obtain checkpoints from Meta / Hugging Face and comply with those terms. This project does not grant rights to redistribute VGGT weights.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data privacy

Original egocentric footage, street scenes, and participant-identifying material are **not** included. Do not commit raw videos, unredacted frames, or large point clouds. Keep private data outside the repository (for example `data/private/`, which is gitignored).

## Current limitations

- Exploratory codebase reconstructed from an earlier application study.
- VGGT reconstructions are up to a similarity (unknown metric scale) unless you supply an external calibration.
- Visibility uses a simplified angular frustum and nearest-neighbor ray hits, not a full renderer.
- Results from any single capture should not be treated as population-level evidence.

Details: [docs/limitations.md](docs/limitations.md).

## Research status

This is an engineering reconstruction of an application pipeline, not a completed validation study and not a claim of a new reconstruction method.
