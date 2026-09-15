# Examples

This phase does **not** ship real egocentric frames.

Private street-level video and participant imagery stay in the original research archive and must not be copied here.

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

Do not rely on even/odd indices.

## Future demo assets

A public demo should use only:

- synthetic scenes, or
- footage you have the right to publish, with faces and identifiers removed

`ANONYMIZED_DEMO_ASSET_REQUIRES_HUMAN_REVIEW`
