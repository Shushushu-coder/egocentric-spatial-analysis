# Data collection principles

These notes are generic. They do not describe a specific capture session, participant, or location.

## Multi-observer egocentric capture

- Use two (or more) wearable cameras looking along a shared path if you intend to compare viewpoints.
- Record overlapping time ranges. A shared audio clap or visible gesture helps later synchronization.
- Prefer matching resolution, lens mode, color profile, and frame rate across cameras. If frame rates differ, align them in preprocessing before paired sampling.
- Keep motion smooth. Fast rotation and motion blur reduce reconstruction quality.
- Include textured surfaces. Blank walls and repeated patterns are poor reconstruction inputs.

## Metric scale (recommended, not implemented as a reliable solver)

Monocular VGGT reconstructions are defined up to scale. If you need meters:

- Place a known-length object fully visible in several frames (checkerboard, tape measure, surveyed baseline).
- Record the true length independently.
- Supply the resulting scale factor to analysis (`--scale`). Do not assume the raw coordinates are meters.

## Privacy

- Treat first-person video as personal data.
- Obtain consent from wearable-camera wearers and anyone you intend to identify.
- Do not publish unredacted street footage, faces, child imagery, or other identifiers with this repository.
- Keep raw media outside git.

## Synchronization budget

Frame-accurate alignment is difficult with independent consumer cameras. Expect residual offset of at least a few frames unless you use genlock or a hardware sync. Document the sync method you used.
