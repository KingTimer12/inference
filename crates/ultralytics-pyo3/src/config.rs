//! Python-facing `InferenceConfig` wrapper.

use pyo3::prelude::*;
use ultralytics_inference::InferenceConfig;

/// Inference configuration exposed to Python.
#[pyclass(name = "InferenceConfig", from_py_object)]
#[derive(Clone)]
pub struct PyInferenceConfig {
    inner: InferenceConfig,
}

#[pymethods]
impl PyInferenceConfig {
    #[new]
    #[pyo3(signature = (
        confidence_threshold = 0.25,
        iou_threshold = 0.45,
        max_det = 300,
        half = false,
        save = true,
        save_frames = false,
        rect = true,
        imgsz = None,
        num_threads = 0,
        batch = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        confidence_threshold: f32,
        iou_threshold: f32,
        max_det: usize,
        half: bool,
        save: bool,
        save_frames: bool,
        rect: bool,
        imgsz: Option<(usize, usize)>,
        num_threads: usize,
        batch: Option<usize>,
    ) -> Self {
        let mut cfg = InferenceConfig::default();
        cfg.confidence_threshold = confidence_threshold;
        cfg.iou_threshold = iou_threshold;
        cfg.max_det = max_det;
        cfg.half = half;
        cfg.save = save;
        cfg.save_frames = save_frames;
        cfg.rect = rect;
        cfg.imgsz = imgsz;
        cfg.num_threads = num_threads;
        cfg.batch = batch;
        Self { inner: cfg }
    }
}

impl From<PyInferenceConfig> for InferenceConfig {
    fn from(py: PyInferenceConfig) -> Self {
        py.inner
    }
}
