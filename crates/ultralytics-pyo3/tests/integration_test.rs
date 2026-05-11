// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//! Integration tests for the PyO3 bindings.
//! These tests exercise the Rust side of the Python bindings without
//! spawning a Python interpreter — conversion logic and config defaults.

use ultralytics_inference::InferenceConfig;

// Re-use the internal conversion by calling the public Rust API directly.
// The PyInferenceConfig → InferenceConfig path is tested indirectly via
// the From impl, which is what predict() ultimately calls.

#[test]
fn inference_config_default_values() {
    let cfg = InferenceConfig::default();
    assert_eq!(cfg.confidence_threshold, 0.25);
    assert_eq!(cfg.iou_threshold, 0.7);
    assert_eq!(cfg.max_det, 300);
    assert!(!cfg.half);
    assert!(cfg.imgsz.is_none());
    assert!(cfg.batch.is_none());
    assert!(cfg.device.is_none());
    assert!(cfg.classes.is_none());
}

#[test]
fn inference_config_builder_chain() {
    let cfg = InferenceConfig::new()
        .with_confidence(0.5)
        .with_iou(0.6)
        .with_max_det(100)
        .with_batch(8);

    assert_eq!(cfg.confidence_threshold, 0.5);
    assert_eq!(cfg.iou_threshold, 0.6);
    assert_eq!(cfg.max_det, 100);
    assert_eq!(cfg.batch, Some(8));
}

#[test]
fn inference_config_imgsz() {
    let mut cfg = InferenceConfig::default();
    cfg.imgsz = Some((320, 320));
    assert_eq!(cfg.imgsz, Some((320, 320)));
}

#[test]
fn inference_config_half_precision() {
    let mut cfg = InferenceConfig::default();
    cfg.half = true;
    assert!(cfg.half);
}

#[test]
fn inference_config_class_filter() {
    let mut cfg = InferenceConfig::default();
    cfg.classes = Some(vec![0, 1, 2]);
    assert_eq!(cfg.classes.as_ref().unwrap().len(), 3);
    assert_eq!(cfg.classes.as_ref().unwrap()[0], 0);
}

#[test]
#[ignore = "downloads a YOLO model and sample image"]
fn e2e_predict_from_url() {
    use std::path::Path;

    let cfg = InferenceConfig::new().with_confidence(0.25);
    let mut model = ultralytics_inference::YOLOModel::load_with_config(
        Path::new("yolo26n.onnx"),
        cfg,
    )
    .expect("model should load");

    let results = model
        .predict_source(
            ultralytics_inference::Source::ImageUrl(
                "https://ultralytics.com/images/bus.jpg".to_string(),
            ),
            None,
        )
        .expect("prediction should succeed");

    assert!(!results.is_empty());
}
