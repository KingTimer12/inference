//! Python-facing source metadata wrapper.

use pyo3::prelude::*;

/// Metadata for a single frame/source returned alongside inference results.
#[pyclass(name = "SourceMeta")]
pub struct PySourceMeta {
    #[pyo3(get)]
    frame_idx: usize,
    #[pyo3(get)]
    total_frames: Option<usize>,
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    fps: Option<f32>,
}

impl PySourceMeta {
    /// Wrap a `SourceMeta` value.
    pub fn wrap(m: ultralytics_inference::SourceMeta) -> Self {
        Self {
            frame_idx: m.frame_idx,
            total_frames: m.total_frames,
            path: m.path,
            fps: m.fps,
        }
    }
}
