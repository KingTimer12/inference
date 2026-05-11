<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Tests Directory (`tests/`)

This directory contains Python integration tests for the `ultralytics-inference` Python package. Tests cover the
Pythonic `YOLO` wrapper, config construction, and source routing logic. End-to-end tests that download a model or sample
image are marked with `@pytest.mark.skip` and must be opted into explicitly.

## 🧪 Overview

- Uses [pytest](https://docs.pytest.org/) as the test runner.
- Unit tests mock the Rust backend so no model file is required.
- End-to-end tests in `TestE2E` require the package to be built (`maturin develop`) and network access.

## 🚀 Running Tests

Build the extension first:

```bash
cd py-inference
maturin develop
```

Run the full suite (skips E2E tests):

```bash
pytest tests/
```

Run end-to-end tests that require network access (downloads model + image):

```bash
pytest tests/ -m "not skip" --runxfail
# or opt-in explicitly:
pytest tests/test_yolo.py::TestE2E --no-header -rN
```

## ✨ Contributing

We love contributions! If you find an issue or have an idea for improving the tests, please open an issue or submit a pull
request. See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for details.
