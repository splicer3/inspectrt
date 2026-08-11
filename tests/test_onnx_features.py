from pathlib import Path

import onnx
from onnx import TensorProto
from onnx.external_data_helper import uses_external_data
import pytest
import torch
from torch import nn

from inspectrt.features import (
    build_resnet50_layer2_extractor,
    extract_patch_embeddings,
)
from inspectrt.onnx_features import build_onnx_feature_graph


@pytest.fixture(scope="module")
def graph_and_reference() -> tuple[nn.Module, nn.Module]:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        reference = build_resnet50_layer2_extractor(weights=None)
        torch.manual_seed(0)
        graph = build_onnx_feature_graph(weights=None)
    return graph, reference


@pytest.fixture(scope="module")
def exported_model(
    tmp_path_factory: pytest.TempPathFactory,
    graph_and_reference: tuple[nn.Module, nn.Module],
) -> tuple[onnx.ModelProto, Path]:
    graph, _ = graph_and_reference
    directory = tmp_path_factory.mktemp("onnx-feature-graph")
    model_path = directory / "model.onnx"
    images = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    program = torch.onnx.export(
        graph,
        (images,),
        f=None,
        dynamo=True,
        input_names=["images"],
        output_names=["layer2", "patch_embeddings"],
        opset_version=20,
        external_data=False,
        dynamic_shapes=None,
        optimize=True,
        verify=False,
    )
    assert isinstance(program, torch.onnx.ONNXProgram)
    program.save(model_path, external_data=False)
    onnx.checker.check_model(model_path, full_check=True)
    return onnx.load(model_path, load_external_data=False), model_path


def test_builds_frozen_evaluation_graph(
    graph_and_reference: tuple[nn.Module, nn.Module],
) -> None:
    graph, _ = graph_and_reference
    parameters = list(graph.parameters())

    assert not graph.training
    assert all(not module.training for module in graph.modules())
    assert parameters
    assert all(not parameter.requires_grad for parameter in parameters)


def test_eager_dual_outputs_exactly_match_the_existing_path(
    graph_and_reference: tuple[nn.Module, nn.Module],
) -> None:
    graph, reference = graph_and_reference
    images = torch.linspace(
        -1.0,
        1.0,
        3 * 256 * 256,
        dtype=torch.float32,
    ).reshape(1, 3, 256, 256)

    with torch.inference_mode():
        expected_layer2 = reference(images)["layer2"]
        expected_patches = extract_patch_embeddings(reference, images)
        layer2, patch_embeddings = graph(images)
        gradient_recording_disabled = not torch.is_grad_enabled()

    assert gradient_recording_disabled
    assert torch.equal(layer2, expected_layer2)
    assert torch.equal(patch_embeddings, expected_patches)
    assert layer2.shape == (1, 512, 32, 32)
    assert patch_embeddings.shape == (1, 1024, 512)
    assert layer2.dtype == patch_embeddings.dtype == torch.float32
    assert layer2.is_contiguous()
    assert patch_embeddings.is_contiguous()
    assert not layer2.requires_grad
    assert not patch_embeddings.requires_grad


def test_exports_static_fp32_dual_output_contract(
    exported_model: tuple[onnx.ModelProto, Path],
) -> None:
    model, _ = exported_model

    assert [_value_info(value) for value in model.graph.input] == [
        ("images", TensorProto.FLOAT, (1, 3, 256, 256))
    ]
    assert [_value_info(value) for value in model.graph.output] == [
        ("layer2", TensorProto.FLOAT, (1, 512, 32, 32)),
        ("patch_embeddings", TensorProto.FLOAT, (1, 1024, 512)),
    ]
    assert [(opset.domain, opset.version) for opset in model.opset_import] == [("", 20)]
    assert {node.domain for node in model.graph.node} == {""}
    assert not model.functions


def test_exports_frozen_pool_and_row_major_layout_without_external_data(
    exported_model: tuple[onnx.ModelProto, Path],
) -> None:
    model, model_path = exported_model
    nodes_by_operation = {
        operation: [node for node in model.graph.node if node.op_type == operation]
        for operation in ("AveragePool", "Transpose", "Reshape")
    }
    assert {
        operation: len(nodes) for operation, nodes in nodes_by_operation.items()
    } == {
        "AveragePool": 1,
        "Transpose": 1,
        "Reshape": 1,
    }

    average_pool = nodes_by_operation["AveragePool"][0]
    transpose = nodes_by_operation["Transpose"][0]
    reshape = nodes_by_operation["Reshape"][0]
    assert _attributes(average_pool) == {
        "auto_pad": b"NOTSET",
        "ceil_mode": 0,
        "count_include_pad": 1,
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
    }
    assert _attributes(transpose) == {"perm": [0, 2, 3, 1]}
    assert average_pool.input == ["layer2"]
    assert transpose.input == average_pool.output
    assert reshape.input[0] == transpose.output[0]
    assert reshape.output == ["patch_embeddings"]
    assert all(
        not uses_external_data(initializer) for initializer in model.graph.initializer
    )
    assert [path.name for path in model_path.parent.iterdir()] == ["model.onnx"]


def _value_info(value: onnx.ValueInfoProto) -> tuple[str, int, tuple[int, ...]]:
    tensor_type = value.type.tensor_type
    dimensions = tensor_type.shape.dim
    assert all(dimension.HasField("dim_value") for dimension in dimensions)
    return (
        value.name,
        tensor_type.elem_type,
        tuple(dimension.dim_value for dimension in dimensions),
    )


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
