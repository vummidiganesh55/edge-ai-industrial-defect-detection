import torch
import onnx

from src.feature_extractor import ResNet18FeatureExtractor


OUTPUT_PATH = "models/resnet18_feature_extractor.onnx"
DEVICE = torch.device("cpu")


class ResNet18ONNXWrapper(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = ResNet18FeatureExtractor()

    def forward(self, x):
        features = self.model(x)

        layer2 = features["layer2"]
        layer3 = features["layer3"]

        return layer2, layer3


def main():

    print("=" * 60)
    print("ONNX MODEL EXPORT")
    print("=" * 60)

    model = ResNet18ONNXWrapper().to(DEVICE)
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE
    )

    print("\nInput shape:")
    print(dummy_input.shape)

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["layer2", "layer3"],
        opset_version=17,
        dynamo=False,
    )

    print("\nONNX export complete.")
    print(f"Saved: {OUTPUT_PATH}")

    onnx_model = onnx.load(OUTPUT_PATH)

    onnx.checker.check_model(onnx_model)

    print("ONNX validation: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()