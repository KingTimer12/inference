<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🐍 Ultralytics YOLO Python Inference

Python bindings for the [Ultralytics YOLO Rust inference engine](https://github.com/ultralytics/inference). Powered by [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/), this package exposes a Pythonic `YOLO` API backed by the same high-performance Rust core — no PyTorch or TensorFlow required.

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

[![PyPI](https://img.shields.io/pypi/v/ultralytics-inference?logo=pypi&logoColor=white&color=blue)](https://pypi.org/project/ultralytics-inference/)
[![Python](https://img.shields.io/pypi/pyversions/ultralytics-inference?logo=python&logoColor=white)](https://pypi.org/project/ultralytics-inference/)

## ✨ Features

- 🚀 **High Performance** — Rust inference engine with zero-cost abstractions, called directly from Python
- 🎯 **Familiar API** — `YOLO` class matching the [Ultralytics Python package](https://github.com/ultralytics/ultralytics) interface
- 🔧 **Multiple Backends** — CPU, CUDA, TensorRT, CoreML, OpenVINO, and more via ONNX Runtime
- 🖼️ **Multiple Sources** — Images, directories, glob patterns, video files, webcams, streams, and URLs
- ⬇️ **Auto Download** — YOLO11 and YOLO26 ONNX models auto-downloaded when not found locally
- 🪶 **No Heavy Runtime** — No PyTorch, TensorFlow, or Python-based inference stack required

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Rust 1.88+](https://rustup.rs/) (for building from source)
- [Maturin](https://www.maturin.rs/) (`pip install maturin`)

### Installation

```bash
# Install from PyPI (pre-built wheels)
pip install ultralytics-inference

# Or install with uv
uv add ultralytics-inference
```

### Development Install

```bash
git clone https://github.com/ultralytics/inference.git
cd inference/py-inference

# Create and activate a virtual environment
uv sync
# or: python -m venv .venv && source .venv/bin/activate

# Build and install the Rust extension in development mode
maturin develop
```

### Export a YOLO Model to ONNX

```bash
# Using Ultralytics CLI
yolo export model=yolo26n.pt format=onnx

# Or with Python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.export(format="onnx")
```

### Run Inference

```python
from ultralytics_inference import YOLO

# Load a model (auto-downloads yolo26n.onnx if not found locally)
model = YOLO("yolo26n.onnx")

# Single image
results = model.predict("image.jpg")

# Remote URL
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.25)

# Directory
results = model.predict("images/")

# Glob pattern
results = model.predict("images/*.jpg")

# Batch (list of paths)
results = model.predict(["img1.jpg", "img2.jpg"])

# Video file
results = model.predict("video.mp4")

# Webcam (device index)
results = model.predict("0")  # or use predict_webcam directly
```

### Example Output

```python
results = model.predict("https://ultralytics.com/images/bus.jpg", conf=0.25, imgsz=640)

for r in results:
    print(r)                  # Results(path='bus.jpg', orig_shape=(1080, 810), boxes=4)
    print(r.orig_shape)       # (1080, 810)
    print(r.names)            # {0: 'person', 5: 'bus', ...}
    print(r.speed)            # (preprocess_ms, inference_ms, postprocess_ms)

    if r.boxes:
        for box in r.boxes:
            x1, y1, x2, y2, conf, cls = box
            print(f"  {r.names[int(cls)]} {conf:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

## 📚 Usage

### `YOLO` Class

```python
from ultralytics_inference import YOLO

model = YOLO("yolo26n.onnx")
```

### `predict()` Method

```python
results = model.predict(
    source,           # str or list[str]
    conf=0.25,        # confidence threshold
    iou=0.45,         # IoU threshold for NMS
    imgsz=None,       # int or (h, w) tuple; None = use model metadata
    half=False,       # FP16 inference
    max_det=300,      # maximum detections per image
    save=False,       # save annotated results
    save_frames=False,# save individual frames for video
    rect=True,        # rectangular inference (minimal padding)
    num_threads=0,    # ONNX Runtime threads (0 = auto)
    batch=None,       # batch size override
)
```

**Source routing:**

| Source value                           | Backend method called  |
| -------------------------------------- | ---------------------- |
| `"image.jpg"` (file path)              | `predict(path)`        |
| `["a.jpg", "b.jpg"]` (list)            | `predict_batch(paths)` |
| `"http://…"` / `"https://…"`           | `predict_url(url)`     |
| `"rtsp://…"` / `"rtmp://…"`            | `predict_url(url)`     |
| `"images/*.jpg"` (glob)                | `predict_glob(pattern)`|
| `"images/"` (directory)                | `predict_dir(dir)`     |

### `Results` Object

Each element of the returned list exposes:

| Attribute     | Type                                    | Description                                |
| ------------- | --------------------------------------- | ------------------------------------------ |
| `.path`       | `str`                                   | Source path / identifier                   |
| `.orig_shape` | `(int, int)`                            | Original image `(height, width)`           |
| `.boxes`      | `list[list[float]] \| None`             | `[[x1, y1, x2, y2, conf, cls], …]`         |
| `.masks`      | `list[list[float]] \| None`             | Segmentation mask rows                     |
| `.probs`      | `list[float] \| None`                   | Classification probabilities               |
| `.keypoints`  | `list[list[float]] \| None`             | Pose keypoint rows                         |
| `.obb`        | `list[list[float]] \| None`             | Oriented bounding box rows                 |
| `.speed`      | `(float, float, float)`                 | `(preprocess_ms, inference_ms, postprocess_ms)` |
| `.names`      | `dict[int, str]`                        | Class ID → name mapping                    |

## 🗂️ Project Structure

```text
py-inference/
├── src/
│   └── ultralytics_inference/
│       ├── __init__.py         # Exposes YOLO
│       └── yolo.py             # YOLO wrapper + source routing
├── tests/
│   ├── test_config.py          # InferenceConfig and routing helper tests
│   ├── test_yolo.py            # YOLO class unit + E2E tests
│   └── README.md
├── pyproject.toml              # Maturin build config + project metadata
└── README.md                   # This file
```

The Rust extension (`ultralytics_pyo3`) lives in [`../crates/ultralytics-pyo3`](../crates/ultralytics-pyo3) and is built by Maturin at install time.

## ⚡ Hardware Acceleration

Hardware acceleration is controlled via `InferenceConfig.device` or passed through the Rust build features. Build the extension with the desired feature:

```bash
# NVIDIA CUDA
maturin develop --features cuda

# Apple CoreML
maturin develop --features coreml

# Intel OpenVINO
maturin develop --features openvino
```

Refer to the [root README](../README.md#-hardware-acceleration) for the full feature list.

## 🧪 Testing

```bash
# Build the extension first
maturin develop

# Run all tests (E2E tests are skipped by default)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run E2E tests (requires network — downloads model + sample image)
pytest tests/test_yolo.py::TestE2E --no-header
```

## 📦 Dependencies

| Package              | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `maturin`            | Build system: compiles and packages the Rust extension |
| `ultralytics_pyo3`   | Auto-built Rust extension (PyO3 bindings)    |

No PyTorch, TensorFlow, or other heavy ML frameworks required.

## 💡 Contributing

Ultralytics thrives on community collaboration, and we deeply value your contributions! Whether it's reporting bugs,
suggesting features, or submitting code changes, your involvement is crucial.

- **Report Issues**: Found a bug? [Open an issue](https://github.com/ultralytics/inference/issues)
- **Feature Requests**: Have an idea? [Share it](https://github.com/ultralytics/inference/issues)
- **Pull Requests**: Read our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) first

## 📜 License

Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is perfect for students, researchers, and enthusiasts. See the [LICENSE](https://github.com/ultralytics/inference/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: For commercial use. Contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📮 Contact

- **GitHub Issues**: [Bug reports and feature requests](https://github.com/ultralytics/inference/issues)
- **Discord**: [Join our community](https://discord.com/invite/ultralytics)
- **Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
