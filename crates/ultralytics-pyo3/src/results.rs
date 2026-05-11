//! Python-facing inference results wrapper.

use pyo3::prelude::*;
use ultralytics_inference::Results;

#[pyclass(name = "Results")]
/// Inference results for a single image.
pub struct PyResults {
    inner: Results,
}

#[pymethods]
impl PyResults {
    #[getter]
    fn path(&self) -> &str {
        &self.inner.path
    }

    #[getter]
    fn orig_shape(&self) -> (u32, u32) {
        self.inner.orig_shape
    }

    /// Boxes as list of [x1, y1, x2, y2, conf, cls] per detection.
    #[getter]
    fn boxes(&self) -> Option<Vec<Vec<f32>>> {
        self.inner
            .boxes
            .as_ref()
            .map(|b| b.data.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Segmentation masks data as flat rows.
    #[getter]
    fn masks(&self) -> Option<Vec<Vec<f32>>> {
        self.inner
            .masks
            .as_ref()
            .map(|m| m.data.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Classification probabilities as list of (class_id, score).
    #[getter]
    fn probs(&self) -> Option<Vec<f32>> {
        self.inner.probs.as_ref().map(|p| p.data.to_vec())
    }

    /// Keypoints as flat rows.
    #[getter]
    fn keypoints(&self) -> Option<Vec<Vec<f32>>> {
        self.inner
            .keypoints
            .as_ref()
            .map(|k| k.data.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// OBB as flat rows.
    #[getter]
    fn obb(&self) -> Option<Vec<Vec<f32>>> {
        self.inner
            .obb
            .as_ref()
            .map(|o| o.data.rows().into_iter().map(|r| r.to_vec()).collect())
    }

    /// Speed as (preprocess_ms, inference_ms, postprocess_ms).
    #[getter]
    fn speed(&self) -> (f64, f64, f64) {
        let s = &self.inner.speed;
        (
            s.preprocess.unwrap_or(0.0),
            s.inference.unwrap_or(0.0),
            s.postprocess.unwrap_or(0.0),
        )
    }

    #[getter]
    fn names(&self) -> std::collections::HashMap<usize, String> {
        self.inner.names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Results(path={:?}, orig_shape={:?}, boxes={})",
            self.inner.path,
            self.inner.orig_shape,
            self.inner.boxes.as_ref().map_or(0, |b| b.data.nrows()),
        )
    }
}

impl PyResults {
    /// Wrap a `Results` value.
    pub fn wrap(r: Results) -> Self {
        Self { inner: r }
    }
}
