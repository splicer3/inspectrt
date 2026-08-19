# Third-party notices and distribution boundary

InspectRT-authored work is provided under the Apache License 2.0; see
[LICENSE](LICENSE). [NOTICE](NOTICE) records the first-party attribution.
Dependencies and external materials retain the terms set by their publishers
and providers.

## Separately installed runtime dependencies

A package manager installs compatible runtime dependencies alongside InspectRT.
Their license files and notices apply to those distributions.

| Direct requirement | Project terms and relevant boundary |
| --- | --- |
| `numpy>=2.4.6` | NumPy publishes [composite project metadata](https://github.com/numpy/numpy/blob/main/pyproject.toml); binary wheels can contain additional bundled material and notices. |
| `pillow>=12.3.0` | Pillow publishes its [MIT-CMU license](https://github.com/python-pillow/Pillow/blob/main/LICENSE). |
| `scikit-learn>=1.9.0` | scikit-learn publishes [BSD-3-Clause terms](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING). Its material SciPy dependency has a separate [license and bundled-material boundary](https://github.com/scipy/scipy/blob/main/LICENSE.txt). |
| `torch>=2.13.0` | PyTorch publishes [composite package metadata](https://github.com/pytorch/pytorch/blob/main/pyproject.toml). Platform-specific binary dependency graphs can include separately licensed components, including packages governed by NVIDIA terms. |
| `torchvision>=0.28.0` | torchvision code is [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE); pretrained weights can have separate terms derived from their training data. |

## Optional ONNX dependencies

The `onnx` extra installs these packages separately:

| Direct requirement | Project terms and relevant boundary |
| --- | --- |
| `onnx>=1.22.0` | ONNX is [Apache-2.0](https://github.com/onnx/onnx/blob/main/LICENSE) and publishes an upstream [NOTICE](https://github.com/onnx/onnx/blob/main/NOTICE). |
| `onnxruntime==1.28.0` | ONNX Runtime 1.28.0 is [MIT](https://github.com/microsoft/onnxruntime/blob/v1.28.0/LICENSE) and carries [third-party notices](https://github.com/microsoft/onnxruntime/blob/v1.28.0/ThirdPartyNotices.txt). InspectRT selects the CPU package. |
| `onnxscript>=0.7.1` | ONNXScript is [MIT](https://github.com/microsoft/onnxscript/blob/main/LICENSE). |

## Build dependency

Hatchling `>=1.27.0` is installed separately for isolated builds and uses the
[MIT License](https://github.com/pypa/hatch/blob/master/LICENSE.txt).

## Datasets, pretrained weights, and generated artifacts

[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) is
obtained separately by the user and retains its CC BY-NC-SA 4.0 terms. MVTec
images and masks stay in the user's data root. Derived embeddings, memory
banks, maps, real fixtures, accepted runs, and screenshots stay in the user's
output locations.

The ResNet-50 `IMAGENET1K_V2` pretrained weights are downloaded and cached
separately through the official torchvision mechanism. Generated ONNX models
store exported parameter values in user-created model files.

## First-party fixture and compact evidence

The canonical retrieval fixture consists entirely of first-party deterministic
synthetic tensors and metadata created for InspectRT.

The repository's compact scientific JSON and numerical SVG files are aggregate
evidence records. Their content consists of aggregate JSON values and numerical
SVG paths. Apache-2.0 covers the InspectRT-authored presentation and code;
external source material retains its original terms.
