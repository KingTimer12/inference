//! Loaded YOLO model handle with all predict variants.

use std::path::Path;

use image::load_from_memory;
use pyo3::prelude::*;
use ultralytics_inference::{Source, YOLOModel};

use crate::{helper::to_py_err, results::PyResults, source::PySourceMeta};

#[pyclass]
/// Loaded model ready for inference.
pub struct Prediction {
    model: YOLOModel,
}

impl Prediction {
    /// Wrap a loaded `YOLOModel`.
    pub fn from_model(model: YOLOModel) -> Self {
        Self { model }
    }
}

#[pymethods]
impl Prediction {
    /// Infer on a single image file path.
    fn predict(&mut self, path: String) -> PyResult<Vec<PyResults>> {
        self.model
            .predict(Path::new(&path))
            .map(|v| v.into_iter().map(PyResults::wrap).collect())
            .map_err(to_py_err)
    }

    /// Infer on raw image bytes (e.g. `open("img.jpg","rb").read()`).
    fn predict_bytes(&mut self, data: &[u8], path: String) -> PyResult<Vec<PyResults>> {
        let img = load_from_memory(data).map_err(to_py_err)?;
        self.model
            .predict_image(&img, path)
            .map(|v| v.into_iter().map(PyResults::wrap).collect())
            .map_err(to_py_err)
    }

    /// Infer on a list of image file paths (batch).
    fn predict_batch(&mut self, paths: Vec<String>) -> PyResult<Vec<Vec<PyResults>>> {
        let images: Result<Vec<_>, _> = paths.iter().map(|p| image::open(p)).collect();
        let images = images.map_err(to_py_err)?;
        self.model
            .predict_batch(&images, &paths)
            .map(|batches| {
                batches
                    .into_iter()
                    .map(|v| v.into_iter().map(PyResults::wrap).collect())
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on an image URL (http/https).
    fn predict_url(&mut self, url: String) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::ImageUrl(url), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on all images in a directory.
    fn predict_dir(&mut self, dir: String) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::Directory(dir.into()), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on a glob pattern (e.g. `"/images/*.jpg"`).
    fn predict_glob(&mut self, pattern: String) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::Glob(pattern), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on a video file.
    fn predict_video(&mut self, path: String) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::Video(path.into()), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on a webcam by device index.
    fn predict_webcam(&mut self, index: u32) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::Webcam(index), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }

    /// Infer on a stream URL (RTSP/RTMP/HTTP).
    fn predict_stream(&mut self, url: String) -> PyResult<Vec<(PySourceMeta, PyResults)>> {
        self.model
            .predict_source(Source::Stream(url), None)
            .map(|v| {
                v.into_iter()
                    .map(|(m, r)| (PySourceMeta::wrap(m), PyResults::wrap(r)))
                    .collect()
            })
            .map_err(to_py_err)
    }
}
