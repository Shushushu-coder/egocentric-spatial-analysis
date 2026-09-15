# Examples

This repository does **not** ship real egocentric frames or reconstruction products.

Private street-level video, child or adult participant imagery, identifiable objects, and original point clouds stay in the research archive and must not be copied here.

Official VGGT demo media (kitchen, landmark videos, and similar) are VGGT Research Materials. They are not re-hosted in this tree.

`NO_PUBLIC_DEMO_ASSET_ADDED`

If someone later proposes anonymized footage, that still needs an explicit rights check:

`ANONYMIZED_DEMO_ASSET_REQUIRES_HUMAN_REVIEW`

## Expected input layout (synthetic illustration)

Use your own videos. A dual-observer run looks like:

```text
your_capture/
  observer_a.mp4
  observer_b.mp4

outputs/frames/
  observer_a/frame_000_observer_a_12.500s.png
  observer_b/frame_000_observer_b_12.500s.png

outputs/images/          # assembled VGGT input folder
outputs/reconstruction/  # extrinsic.npy, intrinsic.npy, points_3d.npy, ...
outputs/analysis/        # analysis_report.json
```

This tree is a format sketch, not experimental results.

## What you can run instead

- `tests/` generates synthetic images and cameras.
- Point `scripts/preprocess.py` at your own videos using `LABEL=path` arguments.
- After VGGT reconstruction, run `scripts/analyze.py` on the `.npy` directory.

## Filename convention

If you do not pass `--labels` or `--labels-json`, analysis can read labels from names like:

```text
frame_000_observer_a_12.500s.png
frame_000_observer_b_12.500s.png
```

Do not rely on even/odd indices. Labels such as `adult` / `child` are allowed only as caller-chosen names; they are not pipeline defaults.

## Future demo assets

A public demo should use only:

- fully synthetic scenes, or
- footage you have the right to publish, with faces and identifiers removed

Do not add files that look like unpublished study outputs.
