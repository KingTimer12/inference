//! Python bindings for the Ultralytics inference library.

use std::path::Path;

pub(crate) mod config;
pub(crate) mod prediction;
pub(crate) mod results;
pub(crate) mod source;

mod helper;

use pyo3::prelude::*;
use ultralytics_inference::{InferenceConfig, YOLOModel};

use crate::{
    config::PyInferenceConfig, helper::to_py_err, prediction::Prediction, results::PyResults,
    source::PySourceMeta,
};

#[pyclass]
struct Model {
    path: String,
    config: Option<InferenceConfig>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(value: String) -> Self {
        Self {
            path: value,
            config: None,
        }
    }

    fn with_config(&self, config: PyInferenceConfig) -> Self {
        Self {
            path: self.path.clone(),
            config: Some(config.into()),
        }
    }

    /// Load model from disk, returning a `Prediction` handle.
    fn load(&self) -> PyResult<Prediction> {
        let cfg = self.config.clone().unwrap_or_default();
        YOLOModel::load_with_config(Path::new(&self.path), cfg)
            .map(Prediction::from_model)
            .map_err(to_py_err)
    }
}

#[pymodule]
mod ultralytics_pyo3 {
    #[pymodule_export]
    use super::Model;
    #[pymodule_export]
    use super::Prediction;
    #[pymodule_export]
    use super::PyInferenceConfig;
    #[pymodule_export]
    use super::PyResults;
    #[pymodule_export]
    use super::PySourceMeta;
}
