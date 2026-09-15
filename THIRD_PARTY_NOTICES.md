# Third-party notices

This repository contains **original application-layer code** (video preprocessing, reconstruction orchestration, viewpoint-aware analysis) plus instructions to call **external** projects. Those layers are not the same legal object.

## Original code in this repository

Preprocessing, CLI wrappers, analysis, and tests in this tree were written for this project and do **not** copy VGGT source.

A top-level license for that original layer has **not** been chosen.

`LICENSE_DECISION_REQUIRES_HUMAN_REVIEW`

Do **not** copy the VGGT `LICENSE.txt` onto this repository and treat it as covering the wrapper or analysis code. That would mis-label original work as Meta research materials.

Human review still needs to decide:

1. Which license (if any) should apply to the original Python in this tree.
2. Whether any reconstruction-wrapper file is close enough to VGGT usage examples to be treated as a derivative of VGGT “Research Materials.”
3. How redistribution of this repo should tell downstream users they still need a separate VGGT grant for code, demos, and weights.

Until that review, this tree should be treated as all-rights-reserved for original files, with VGGT remaining under its own terms.

## VGGT (code and demonstration materials)

Reconstruction depends on **VGGT** (Visual Geometry Grounded Transformer):

- Paper: Wang, Jianyuan et al. “VGGT: Visual Geometry Grounded Transformer.” CVPR 2025.
- Code: https://github.com/facebookresearch/vggt
- License: Meta’s VGGT License (custom; not MIT / Apache / GPL), including an Acceptable Use Policy.

VGGT grants a limited license to use, reproduce, distribute, and modify its Research Materials, and requires that **distribution of those materials or derivative works of them** stay under the VGGT Agreement, with publication attribution. This repository:

- does not vendor VGGT source
- does not include official VGGT demo images or videos
- cannot grant rights to VGGT materials

Anyone who installs VGGT, redistributes VGGT files, or publishes results that used VGGT must follow the VGGT license and AUP themselves.

## VGGT model weights

Checkpoints (`model.pt` and Hugging Face `facebook/VGGT-1B` or similar) are **not** in this repository. Weight terms can differ from source-code terms (including non-commercial vs commercial checkpoint distinctions, depending on the file you download). Obtain weights from Meta / Hugging Face and read the card and license attached to **that** artifact. This project does not re-host or sublicense weights.

## Other Python dependencies

Runtime libraries (NumPy, OpenCV, SciPy, Matplotlib, and optional PyTorch / trimesh / Hugging Face Hub) remain under their own licenses. Install them from their upstream distributions.
