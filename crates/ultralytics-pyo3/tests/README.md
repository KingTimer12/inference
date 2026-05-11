<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Tests Directory (`tests/`)

This directory contains Rust integration tests for the `ultralytics-pyo3` bindings crate. Tests cover the Rust side of
the Python bindings — config construction, type conversions, and end-to-end inference calls — without requiring a Python
interpreter.

## 🧪 Overview

- Uses standard [Cargo tests](https://doc.rust-lang.org/cargo/guide/tests.html) with Rust's built-in test harness.
- Tests that download a model or sample image are marked `#[ignore]` and must be opted into explicitly.
- Mirrors the pattern established in the `ultralytics-inference` integration tests.

## 🚀 Running Tests

Run the full suite from the project root:

```bash
cargo test --package ultralytics-pyo3
```

Run end-to-end tests that require network access (downloads model + image):

```bash
cargo test --package ultralytics-pyo3 -- --include-ignored
```

To generate code coverage locally (Linux recommended):

```bash
cargo llvm-cov --package ultralytics-pyo3 --html
```

## ✨ Contributing

We love contributions! If you find an issue or have an idea for improving the tests, please open an issue or submit a pull
request. See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details.
