# 第一人称视点空间分析

[English](README.md) | 简体中文

一个用于第一人称（egocentric）视频预处理、基于 VGGT 的三维重建，以及视点感知空间分析的可复用 pipeline。

**输入：** 一段或多段带标签的第一人称视频（或已有重建目录）。
**输出：** 与 VGGT 兼容的帧序列、相机 / 点云数组，以及相对几何与可见性指标的 JSON 报告。

[VGGT](https://github.com/facebookresearch/vggt) 是**外部**重建依赖。本仓库不是新的三维模型，不是 VGGT 的 fork，也不是具备可靠度量尺度的重建框架。

本仓库中由作者编写的工作，是围绕 VGGT 的工程层：多视频处理、视频预处理、重建编排、序列化，以及基于重建产物的视点感知分析。

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

## 可以用这份代码做什么

- 一次运行处理多段带标签视频（`LABEL=path`；身份从不根据偶数/奇数帧顺序推断）
- 探测容器元数据（分辨率、FPS、时长）
- 当码流帧率不同时，用 ffmpeg 做帧率对齐
- 在时间窗口内采样帧（`uniform`、`quality_weighted`、`complementary`）
- 为候选帧打分（锐度、曝光、对比度，可选 ORB；若缺少 OpenCV 特征检测器则回退到 NumPy）
- 缩放 / 裁剪 / 填充为与 VGGT 兼容的正方形输入（默认 518）
- 将带标签的帧文件夹组装成一个重建用图像目录
- 编排 VGGT 推理，支持设备选择与内存上限（`--max-images`）
- 序列化相机、深度与点（`extrinsic.npy`、`intrinsic.npy`、`points_3d.npy`，可选 PLY）
- 分析相机几何（OpenCV `[R|t]`、观察角度）
- 比较视点（仰角 / 偏航类角度）
- 相对 up-axis 百分位“地面”报告相对高度
- 运行简化的视锥体式可见性
- 用 KD-tree 近似射线邻近（不是网格光线追踪）

除非传入外部 `--scale`，距离均为**重建坐标单位**。不默认假设度量尺度。

## 仓库结构

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

## 安装

需要 Python 3.10+。

```bash
pip install -r requirements.txt
pip install -e .
```

视频预处理需要**可用的** OpenCV 4.x 构建，并提供 `cv2.VideoCapture` / `cv2.VideoWriter`。请只声明**一种** wheel：

- `opencv-python>=4.8,<5`（`requirements.txt` 中的默认项），或
- 同一 4.x 范围内的 `opencv-python-headless`

不要同时安装两者，也不要混入 `opencv-contrib-python`。OpenCV 5.x 会拉取 `numpy>=2`，与本项目的 `numpy>=1.24,<2` 冲突。残留的空 `cv2/` 目录或损坏的 contrib 安装会遮蔽真正的包：`import cv2` 成功，但缺少视频 I/O 属性，视频测试会被跳过。质量指标仍可通过 NumPy 回退运行；帧提取则不行。

`pip install -r requirements.txt` 或 `pip install -e .` **不会**安装 VGGT 重建。这些命令也不会下载 `model.pt`。请从上游安装官方 VGGT 包，并自行获取有许可的 checkpoint。本仓库不包含 VGGT 源码或权重。

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

请将 torch / CUDA 与本机环境匹配。重建 extras 不能替代 VGGT 安装。

## 基本用法

预处理带标签的观察者视频：

```bash
python scripts/preprocess.py extract \
  --video observer_a=path/to/a.mp4 \
  --video observer_b=path/to/b.mp4 \
  --start-time 10 --end-time 20 \
  --num-frames 10 \
  --strategy quality_weighted \
  --output-dir outputs/frames
```

缩放并组装重建文件夹：

```bash
python scripts/preprocess.py resize --input-dir outputs/frames/observer_a --output-dir outputs/images --size 518
python scripts/preprocess.py assemble \
  --frames observer_a=outputs/frames/observer_a \
  --frames observer_b=outputs/frames/observer_b \
  --output-dir outputs/images
```

重建（需要本地 VGGT 安装与权重）：

```bash
python scripts/reconstruct.py \
  --image-dir outputs/images \
  --output-dir outputs/reconstruction \
  --weight-path path/to/VGGT/model.pt \
  --device auto \
  --max-images 20
```

使用显式标签进行分析，或使用形如 `frame_000_<label>_...` 的文件名：

```bash
python scripts/analyze.py \
  --reconstruction-dir outputs/reconstruction \
  --output-dir outputs/analysis \
  --labels observer_a observer_b observer_a observer_b
```

### 测试

```bash
pip install -r requirements-dev.txt
pytest
```

几何、采样与分析测试使用合成数组。视频提取测试会用 OpenCV `VideoWriter` 写入一段极小的合成片段，再用 `VideoCapture` 读取（无私有影像，不模拟 capture）。在可用的 OpenCV 4.x wheel 下，该测试应实际运行，而不是被跳过。若 `cv2` 缺少这些属性，该测试会**跳过**，而不是伪装成通过。

## VGGT 依赖与致谢

重建使用 **VGGT: Visual Geometry Grounded Transformer**（Wang et al., CVPR 2025）。请安装并引用官方项目。代码、演示材料与 checkpoint 仍受 VGGT 许可与可接受使用政策约束。用户必须从 Meta / Hugging Face 获取 checkpoint，并遵守相应条款。本项目不授予对 VGGT 源码或权重的权利。

见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本仓库中的原创代码被归类为独立 VGGT 包装层（`INDEPENDENT_WRAPPER`），在技术上已可另行采用顶层许可证（`READY_FOR_INDEPENDENT_TOP_LEVEL_LICENSE`）。目前尚未包含 SPDX `LICENSE` 文件；请另行选择 MIT / Apache-2.0 / BSD（或其他许可证）。不要把 VGGT 许可视为覆盖包装层、视频预处理或分析层。请保留 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 数据隐私

原始第一人称影像、可识别人物以及与参与者相关的文件**均未包含**。请勿提交原始视频、未脱敏帧或大型点云。将私有数据放在仓库之外（例如已被 gitignore 的 `data/private/`）。

## 限制

- VGGT 是外部依赖；重建质量与许可属于上游问题。
- 不假设度量尺度。默认分析聚焦相对量（角度、比例、可见性分数）。
- “地面”是 up-axis 百分位，不是拟合平面。
- 可见性是简化的角向视锥体；射线命中是 KD-tree 邻近，而不是渲染器或网格光线求交。
- 这是工程 / 研究 pipeline，不是已验证的人群层面研究。
- 私有研究影像已排除，因此该影像上未发表的数值无法在此复现。

细节见 [docs/limitations.md](docs/limitations.md)。

## 项目起源 / 探索性案例

该 pipeline 最初围绕一次私有双可穿戴相机采集搭建，其中包括一次探索性的成人/儿童街景比较。该会话**仅作为历史背景**。它不是已完成的验证研究，不是关于儿童空间感知的结论，也不构成对 API 的约束（观察者始终由调用方提供标签）。
