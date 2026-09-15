# Third-party notices

This repository contains **original application-layer code** (video preprocessing, reconstruction orchestration, viewpoint-aware analysis) plus instructions to call **external** projects. Those layers are not the same legal object.

## Original code in this repository

Preprocessing, CLI wrappers, analysis, and tests in this tree were written for this project and do **not** copy VGGT source files, weights, or official demo media.

Technical classification after a source-tree review (this is not legal advice):

- `INDEPENDENT_WRAPPER` — `src/egocentric_spatial_analysis/reconstruction/backend.py` calls VGGT’s public Python API (`vggt.models.vggt.VGGT`, `load_and_preprocess_images_square`, `pose_encoding_to_extri_intri`, `unproject_depth_map_to_point_map`, plus `aggregator` / `camera_head` / `depth_head`) and owns orchestration, device selection, numpy/PLY serialization, and error handling. It does not vendor VGGT modules or rewrite VGGT internals.
- `READY_FOR_INDEPENDENT_TOP_LEVEL_LICENSE` — original files in this repository can carry their own license, independent of VGGT.

This notice does **not** choose MIT / Apache-2.0 / BSD (or any other SPDX license). Add a top-level `LICENSE` only after a human picks one. Until that file exists, treat original files as all-rights-reserved except as you later grant.

Do **not** copy the VGGT `LICENSE.txt` onto this repository and treat it as covering the wrapper, preprocessing, or analysis layers.

VGGT remains under its own third-party terms. Keep this file. README must keep attribution and the external-dependency statement. Downstream users still need a separate VGGT grant for VGGT code, demos, and weights.

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
