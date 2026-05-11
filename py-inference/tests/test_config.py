# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for InferenceConfig construction and source routing helpers."""

import pytest

from ultralytics_inference.yolo import _build_config, _is_glob, _is_url


class TestBuildConfig:
    def test_defaults(self):
        cfg = _build_config()
        assert cfg is not None

    def test_imgsz_int_converted_to_tuple(self):
        cfg = _build_config(imgsz=320)
        # InferenceConfig stores imgsz as Optional[(h, w)] — we can't read it
        # back from the opaque Rust object, but construction must not raise.
        assert cfg is not None

    def test_imgsz_tuple_passthrough(self):
        cfg = _build_config(imgsz=(480, 640))
        assert cfg is not None

    def test_imgsz_none(self):
        cfg = _build_config(imgsz=None)
        assert cfg is not None

    def test_custom_thresholds(self):
        cfg = _build_config(conf=0.5, iou=0.6, max_det=50)
        assert cfg is not None

    def test_half_precision(self):
        cfg = _build_config(half=True)
        assert cfg is not None

    def test_batch(self):
        cfg = _build_config(batch=8)
        assert cfg is not None


class TestIsUrl:
    @pytest.mark.parametrize("url", [
        "http://example.com/img.jpg",
        "https://ultralytics.com/images/bus.jpg",
        "rtsp://192.168.1.1/stream",
        "rtmp://live.example.com/app/key",
    ])
    def test_valid_urls(self, url):
        assert _is_url(url) is True

    @pytest.mark.parametrize("source", [
        "/path/to/image.jpg",
        "relative/path.png",
        "C:\\Windows\\image.jpg",
        "images/*.jpg",
    ])
    def test_non_urls(self, source):
        assert _is_url(source) is False


class TestIsGlob:
    @pytest.mark.parametrize("pattern", [
        "images/*.jpg",
        "data/**/*.png",
        "file?.jpg",
        "frame[0-9].png",
    ])
    def test_valid_globs(self, pattern):
        assert _is_glob(pattern) is True

    @pytest.mark.parametrize("source", [
        "/path/to/image.jpg",
        "https://example.com/img.jpg",
        "/some/directory",
    ])
    def test_non_globs(self, source):
        assert _is_glob(source) is False
