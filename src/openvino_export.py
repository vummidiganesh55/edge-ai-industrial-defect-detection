from pathlib import Path

import openvino as ov


ONNX_PATH = "models/resnet18_feature_extractor.onnx"
OUTPUT_DIR = Path("models/openvino")


def main():

    print("=" * 60)
    print("OPENVINO MODEL CONVERSION")
    print("=" * 60)

    print(f"Input ONNX : {ONNX_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    core = ov.Core()

    print("\nReading ONNX model...")

    model = core.read_model(
        ONNX_PATH
    )

    print("Converting to OpenVINO IR...")

    ov.serialize(
        model,
        str(
            OUTPUT_DIR /
            "resnet18_feature_extractor.xml"
        ),
        str(
            OUTPUT_DIR /
            "resnet18_feature_extractor.bin"
        )
    )

    print("\nOpenVINO conversion complete.")

    print(
        f"XML: {OUTPUT_DIR / 'resnet18_feature_extractor.xml'}"
    )

    print(
        f"BIN: {OUTPUT_DIR / 'resnet18_feature_extractor.bin'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()