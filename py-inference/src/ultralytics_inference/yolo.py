from __future__ import annotations

import os
from typing import Union

from ultralytics_pyo3 import InferenceConfig, Model, Prediction, Results


def _build_config(
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    half: bool = False,
    save: bool = False,
    save_frames: bool = False,
    rect: bool = True,
    imgsz: Union[int, tuple[int, int], None] = None,
    num_threads: int = 0,
    batch: Union[int, None] = None,
) -> InferenceConfig:
    """Build an InferenceConfig from Pythonic keyword arguments.

    Args:
        conf (float): Confidence threshold for detections (0.0–1.0).
        iou (float): IoU threshold used by Non-Maximum Suppression (0.0–1.0).
        max_det (int): Maximum number of detections returned per image.
        half (bool): Use FP16 half-precision inference. Faster on GPU, slightly
            lower accuracy. Avoid on CPU-only deployments.
        save (bool): Save annotated result images to disk.
        save_frames (bool): When the source is a video, save individual frames
            instead of a single output video file.
        rect (bool): Rectangular inference — pads to the smallest multiple of the
            model stride with minimal border, reducing wasted computation.
        imgsz (int | tuple[int, int] | None): Inference resolution. A bare ``int``
            is expanded to ``(int, int)``. ``None`` reads the size from the
            model's ONNX metadata.
        num_threads (int): ONNX Runtime intra-op thread count. ``0`` lets ORT
            pick the optimal value automatically (recommended).
        batch (int | None): Batch size override. ``None`` defaults to 1 for
            single-image inference.

    Returns:
        (InferenceConfig): Configured object ready to pass to Model.with_config.

    Examples:
        >>> cfg = _build_config(conf=0.5, imgsz=320)
        >>> cfg = _build_config(imgsz=(480, 640), half=True)
    """
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    return InferenceConfig(
        confidence_threshold=conf,
        iou_threshold=iou,
        max_det=max_det,
        half=half,
        save=save,
        save_frames=save_frames,
        rect=rect,
        imgsz=imgsz,
        num_threads=num_threads,
        batch=batch,
    )


def _is_url(source: str) -> bool:
    """Return True for HTTP, HTTPS, RTSP, and RTMP source strings.

    Args:
        source (str): Source string to test.

    Returns:
        (bool): True if source is a network URL, False otherwise.

    Examples:
        >>> _is_url("https://ultralytics.com/images/bus.jpg")  # True
        >>> _is_url("images/bus.jpg")  # False
    """
    return (
        source.startswith("http://")
        or source.startswith("https://")
        or source.startswith("rtsp://")
        or source.startswith("rtmp://")
    )


def _is_glob(source: str) -> bool:
    """Return True when source contains shell glob meta-characters.

    Args:
        source (str): Source string to test.

    Returns:
        (bool): True if source contains ``*``, ``?``, or ``[``.

    Examples:
        >>> _is_glob("images/*.jpg")  # True
        >>> _is_glob("images/bus.jpg")  # False
    """
    return any(c in source for c in ("*", "?", "["))


class YOLO:
    """High-level YOLO interface backed by the Rust ultralytics-inference engine.

    The model file is registered on construction but not loaded from disk until
    predict is called, keeping __init__ side-effect free.

    Attributes:
        _path (str): Path to the ONNX model file.
        _model (Model): Underlying Rust Model handle.

    Examples:
        >>> from ultralytics_inference import YOLO
        >>> model = YOLO("yolo26n.onnx")
        >>> results = model.predict("https://ultralytics.com/images/bus.jpg", conf=0.25, imgsz=640)
        >>> for r in results:
        ...     print(r)
    """

    def __init__(self, model: str) -> None:
        """Register the model path without loading it.

        Args:
            model (str): Path to a local ``.onnx`` file, or a known model name
                that will be auto-downloaded (e.g. ``"yolo26n.onnx"``).

        Examples:
            >>> model = YOLO("yolo26n.onnx")
        """
        self._path = model
        self._model: Model = Model(model)
        self._prediction: Prediction | None = None

    def _get_prediction(self, cfg: InferenceConfig) -> Prediction:
        """Apply cfg and load the model, returning a Prediction handle.

        Args:
            cfg (InferenceConfig): Inference configuration to apply.

        Returns:
            (Prediction): Loaded model ready for inference calls.
        """
        return self._model.with_config(cfg).load()

    def predict(
        self,
        source: Union[str, list[str]],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Union[int, tuple[int, int], None] = None,
        half: bool = False,
        max_det: int = 300,
        save: bool = False,
        save_frames: bool = False,
        rect: bool = True,
        num_threads: int = 0,
        batch: Union[int, None] = None,
    ) -> list[Results]:
        """Run inference on source and return a flat list of Results.

        Source routing:
            - ``list[str]``       → predict_batch
            - ``http/https`` URL  → predict_url
            - ``rtsp/rtmp`` URL   → predict_url
            - Glob pattern        → predict_glob
            - Directory path      → predict_dir
            - File path           → predict

        Args:
            source (str | list[str]): Input source — file path, URL, glob pattern,
                directory, or list of file paths.
            conf (float): Confidence threshold (0.0–1.0).
            iou (float): IoU threshold for NMS (0.0–1.0).
            imgsz (int | tuple[int, int] | None): Inference size. ``None`` reads
                from model ONNX metadata.
            half (bool): Enable FP16 inference.
            max_det (int): Maximum detections per image.
            save (bool): Save annotated images to disk.
            save_frames (bool): Save individual video frames instead of a video file.
            rect (bool): Enable rectangular (minimal-padding) inference.
            num_threads (int): ONNX Runtime thread count; ``0`` = auto.
            batch (int | None): Batch size override; ``None`` = model default.

        Returns:
            (list[Results]): Flat list of Results, one per image or frame processed.

        Examples:
            >>> model = YOLO("yolo26n.onnx")
            >>> results = model.predict("bus.jpg", conf=0.5)
            >>> results = model.predict("https://ultralytics.com/images/bus.jpg", imgsz=320, save=True)
            >>> results = model.predict(["img1.jpg", "img2.jpg"])
        """
        cfg = _build_config(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            half=half,
            max_det=max_det,
            save=save,
            save_frames=save_frames,
            rect=rect,
            num_threads=num_threads,
            batch=batch,
        )
        pred = self._get_prediction(cfg)

        if isinstance(source, list):
            batches = pred.predict_batch(source)
            return [r for batch in batches for r in batch]

        if _is_url(source):
            pairs = pred.predict_url(source)
            return [r for _, r in pairs]

        if _is_glob(source):
            pairs = pred.predict_glob(source)
            return [r for _, r in pairs]

        if os.path.isdir(source):
            pairs = pred.predict_dir(source)
            return [r for _, r in pairs]

        return pred.predict(source)
