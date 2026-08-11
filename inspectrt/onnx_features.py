"""Static ONNX-exportable frozen feature graph."""

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights

from inspectrt.features import (
    _pool_and_layout_layer2,
    build_resnet50_layer2_extractor,
)


class _OnnxFeatureGraph(nn.Module):
    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        layer2 = self.extractor(images)["layer2"]
        _, patch_embeddings = _pool_and_layout_layer2(layer2)
        return layer2, patch_embeddings


def build_onnx_feature_graph(
    weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
) -> nn.Module:
    """Build the frozen static feature graph used for ONNX export."""
    graph = _OnnxFeatureGraph(build_resnet50_layer2_extractor(weights=weights))
    graph.requires_grad_(False)
    return graph.eval()
