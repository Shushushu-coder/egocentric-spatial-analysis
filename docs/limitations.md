# Limitations

This project is a reusable engineering and analysis pipeline around an external VGGT backend. It is not a finished perceptual study, not a new reconstruction network, and not a validated population-level method.

## Metric-scale recovery

VGGT (like other monocular feed-forward reconstructors) does not provide a reliable absolute metric scale from images alone. Earlier calibration attempts using assumed observer heights, walking distance, or a single ground plane did not recover a consistent similarity transform: VGGT’s internal normalization can stretch axes differently, so one scalar is not always enough.

**Current policy:** analysis defaults to reconstruction units. Angles, ratios, and visibility fractions are the supported comparisons. `--scale` exists only for an **externally** measured factor. Metric reconstruction is **not** a reliable built-in capability.

## Reconstruction

- Quality depends on overlap, texture, motion blur, and how many frames fit in memory.
- The wrapper may drop frames beyond `--max-images`.
- Bundle adjustment is not included.
- Depth confidence thresholding is heuristic.

## Analysis approximations

- “Ground” is a percentile of the up-axis coordinate, not a fitted plane.
- Frustum tests use independent horizontal/vertical angles, not a full camera matrix test.
- Ray hits use KD-tree proximity along sampled rays, not mesh intersections.
- The default up axis is Y; VGGT scenes may use a different convention.

## Software and environment

- VGGT must be installed and licensed separately.
- Tests cover project-owned geometry and preprocessing on synthetic inputs, not VGGT accuracy.
- A complete video-extraction test requires a working OpenCV install that exports `cv2.VideoCapture` and `cv2.VideoWriter`. An empty or broken `cv2` namespace (common with leftover `opencv-contrib-python` folders) causes that test to skip; it is not mocked as a pass.
- Windows-specific hardcoded paths from a prior private codebase were removed; remaining platform differences (ffmpeg on PATH, codecs) still apply.

## Study size and interpretation

An earlier private capture used a short window and a small number of frames. Nothing in this repository should be read as population-level evidence about groups of people, age, or height. Do not treat unpublished numeric ratios from that footage as project claims.

## Privacy and data

Original footage is excluded. Without shared public data, published numbers from that footage cannot be reproduced here.
