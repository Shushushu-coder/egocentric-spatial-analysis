# Methodology

## Pipeline

```text
egocentric video → frame sequence → reconstruction → viewpoint-aware analysis
```

Two or more time-overlapping first-person videos can be processed as labeled observers. Each observer / viewpoint is a named input. The software never assigns identity from frame index parity.

## Preprocessing (project contribution)

1. **Probe** container metadata (resolution, FPS, duration).
2. **Align FPS** when streams were recorded at different rates, using ffmpeg. Alignment is per-file and configured by the caller.
3. **Sample frames** in a time window using `uniform`, `quality_weighted`, or `complementary` strategies. Quality combines Laplacian sharpness, brightness, contrast, and an optional ORB count (NumPy gradient fallback if OpenCV feature detectors are unavailable), with penalties for blur and bad exposure.
4. **Resize** to a square (default 518, VGGT’s patch-aligned input size) via center crop, pad, or stretch.
5. **Assemble** labeled folders into one image directory. Filenames keep the observer label.

Video probing and frame extraction require a working OpenCV `VideoCapture`. Quality scoring of in-memory arrays does not.

## Reconstruction (VGGT capability)

VGGT is an external feed-forward model. Given N images it predicts camera parameters, depth, and point maps. This repository only:

- selects a device
- caps image count for memory
- loads a user-provided checkpoint (or an explicit `from_pretrained` request)
- writes `extrinsic.npy`, `intrinsic.npy`, `points_3d.npy`, depth arrays, and an optional PLY

Camera convention follows OpenCV `[R|t]`, with world camera center `C = -Rᵀ t`.

## Analysis (project contribution)

Inputs are reconstruction arrays plus **explicit labels**.

| Quantity | What is computed | Scale dependence |
|---|---|---|
| Euler angles, elevation | Viewing direction vs a chosen up axis | Angles are scale-invariant |
| Height layers | Histogram of point coordinates on the up axis | Bin widths are reconstruction units |
| Relative height | Share of points below / near / above a camera | Uses reconstruction units; ratios are comparable |
| Ground visibility | Low-percentile “ground” points inside a frustum | Ratios are relative; distances need scale |
| Ray hits | Sampled rays vs a KD-tree of points | Distances need scale |

If the caller passes `--scale`, translations and points are multiplied by that factor. The report then marks units as `metric_if_calibrated`. No internal method currently estimates that factor reliably.

## What is VGGT vs what is this project

- **VGGT:** learned multi-view geometry (cameras, depth, points).
- **This project:** capture-oriented preprocessing, memory-aware wrapping, and labeled viewpoint-aware analysis on those artifacts.

Observer roles such as “adult” or “child” are not built into the method. They are ordinary labels if a caller chooses them.
