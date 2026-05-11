# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for the YOLO Python wrapper."""

from typing import Any

import pytest

from ultralytics_inference import YOLO


class TestYOLOInstantiation:
    def test_yolo_accepts_path_string(self):
        # Construction must not raise even if the model file doesn't exist yet —
        # loading is deferred to predict().
        model = YOLO("yolo26n.onnx")
        assert model._path == "yolo26n.onnx"

    def test_yolo_stores_model_path(self):
        model = YOLO("/tmp/custom_model.onnx")
        assert model._path == "/tmp/custom_model.onnx"


class TestSourceRouting:
    """Test that predict() routes to the right backend method.

    These tests mock _get_prediction so no real model is needed.
    """

    def _make_model(self, monkeypatch, capture: list[tuple[str, Any]]):
        model = YOLO("fake.onnx")

        class FakePrediction:
            def predict(self, path):
                capture.append(("predict", path))
                return []

            def predict_batch(self, paths):
                capture.append(("predict_batch", paths))
                return []

            def predict_url(self, url):
                capture.append(("predict_url", url))
                return []

            def predict_glob(self, pattern):
                capture.append(("predict_glob", pattern))
                return []

            def predict_dir(self, directory):
                capture.append(("predict_dir", directory))
                return []

        monkeypatch.setattr(model, "_get_prediction", lambda cfg: FakePrediction())
        return model

    def test_routes_file_to_predict(self, monkeypatch, tmp_path):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        img = tmp_path / "img.jpg"
        img.touch()
        model.predict(str(img))
        assert calls[0][0] == "predict"

    def test_routes_list_to_predict_batch(self, monkeypatch):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        model.predict(["a.jpg", "b.jpg"])
        assert calls[0][0] == "predict_batch"

    def test_routes_http_url(self, monkeypatch):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        model.predict("https://ultralytics.com/images/bus.jpg")
        assert calls[0][0] == "predict_url"

    def test_routes_rtsp_url(self, monkeypatch):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        model.predict("rtsp://192.168.1.1/stream")
        assert calls[0][0] == "predict_url"

    def test_routes_glob_pattern(self, monkeypatch):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        model.predict("images/*.jpg")
        assert calls[0][0] == "predict_glob"

    def test_routes_directory(self, monkeypatch, tmp_path):
        calls: list[tuple[str, Any]] = []
        model = self._make_model(monkeypatch, calls)
        model.predict(str(tmp_path))
        assert calls[0][0] == "predict_dir"


@pytest.mark.skip(reason="downloads a YOLO model and sample image")
class TestE2E:
    def test_predict_from_url(self, tmp_path):
        model = YOLO(str(tmp_path / "yolo26n.onnx"))
        results = model.predict(
            "https://ultralytics.com/images/bus.jpg",
            conf=0.25,
            imgsz=320,
            save=False,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        first = results[0]
        assert first.orig_shape[0] > 0
        assert first.orig_shape[1] > 0

    def test_predict_with_imgsz_int(self, tmp_path):
        model = YOLO(str(tmp_path / "yolo26n.onnx"))
        results = model.predict(
            "https://ultralytics.com/images/bus.jpg",
            imgsz=320,
        )
        assert len(results) > 0

    def test_predict_results_have_boxes(self, tmp_path):
        model = YOLO(str(tmp_path / "yolo26n.onnx"))
        results = model.predict("https://ultralytics.com/images/bus.jpg")
        detections = [r for r in results if r.boxes is not None]
        assert len(detections) > 0
        boxes = detections[0].boxes
        assert boxes is not None
        assert len(boxes[0]) >= 6  # x1, y1, x2, y2, conf, cls
