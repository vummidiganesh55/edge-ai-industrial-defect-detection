from pathlib import Path

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)

INPUT_MODEL = Path(
    "models/resnet18_feature_extractor.onnx"
)

OUTPUT_MODEL = Path(
    "models/resnet18_feature_extractor_int8.onnx"
)


def main():

    print("=" * 60)
    print("ONNX INT8 QUANTIZATION")
    print("=" * 60)

    print(f"Input model : {INPUT_MODEL}")
    print(f"Output model: {OUTPUT_MODEL}")

    if not INPUT_MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {INPUT_MODEL}"
        )

    OUTPUT_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nQuantizing model...")

    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8,
    )

    print("\nQuantization complete.")
    print(f"Saved: {OUTPUT_MODEL}")

    input_size = INPUT_MODEL.stat().st_size / (1024 * 1024)
    output_size = OUTPUT_MODEL.stat().st_size / (1024 * 1024)

    print("\nMODEL SIZE")
    print("-" * 60)
    print(f"FP32 ONNX : {input_size:.2f} MB")
    print(f"INT8 ONNX : {output_size:.2f} MB")

    if output_size > 0:
        print(
            f"Size ratio: "
            f"{input_size / output_size:.2f}x"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()