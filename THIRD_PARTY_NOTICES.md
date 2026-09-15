# Third-party notices

This repository's original Python code (preprocessing, VGGT wrapper orchestration, and spatial analysis) is a separate application layer. A top-level LICENSE for that layer has **not** been chosen yet.

`LICENSE_DECISION_REQUIRES_HUMAN_REVIEW`

## VGGT

Reconstruction depends on **VGGT** (Visual Geometry Grounded Transformer):

- Paper: Wang, Jianyuan et al. "VGGT: Visual Geometry Grounded Transformer." CVPR 2025.
- Code: https://github.com/facebookresearch/vggt
- Checkpoints: distributed by Meta under the VGGT license and acceptable-use policy (including non-commercial vs commercial checkpoint distinctions).

This repository:

- does not vendor VGGT source
- does not include `model.pt` or other weights
- cannot grant rights to VGGT materials

If you distribute reconstruction outputs or derivative tools that include VGGT code or weights, you must follow the VGGT license, including attribution for research publications.

## Other Python dependencies

Runtime libraries (NumPy, OpenCV, SciPy, Matplotlib, and optional PyTorch / trimesh / Hugging Face Hub) remain under their own licenses. Install them from their upstream distributions.
