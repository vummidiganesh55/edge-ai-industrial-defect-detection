import os

import torch
import tensorrt as trt

from src.feature_extractor import ResNet18FeatureExtractor


FP16_ONNX_PATH = "models/resnet18_feature_extractor_fp16.onnx"
FP16_ENGINE_PATH = "models/resnet18_feature_extractor_fp16.engine"


def export_fp16_onnx():
    print("Exporting FP16 ONNX model...")

    model = (
        ResNet18FeatureExtractor()
        .cuda()
        .half()
        .eval()
    )

    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device="cuda",
        dtype=torch.float16,
    )

    os.makedirs(
        os.path.dirname(FP16_ONNX_PATH),
        exist_ok=True,
    )

    # Remove old ONNX files before exporting.
    # This prevents stale external-data files from causing
    # TensorRT parsing problems.
    if os.path.exists(FP16_ONNX_PATH):
        os.remove(FP16_ONNX_PATH)

    external_data_path = FP16_ONNX_PATH + ".data"

    if os.path.exists(external_data_path):
        os.remove(external_data_path)

    print("Exporting with ONNX opset 18...")
    print("Embedding model weights inside the ONNX file...")

    torch.onnx.export(
        model,
        dummy_input,
        FP16_ONNX_PATH,
        input_names=["input"],
        output_names=["layer2", "layer3"],
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )

    if not os.path.exists(FP16_ONNX_PATH):
        raise RuntimeError(
            "FP16 ONNX file was not created."
        )

    file_size_mb = (
        os.path.getsize(FP16_ONNX_PATH)
        / (1024 * 1024)
    )

    print()
    print("FP16 ONNX created successfully.")
    print("Output:", FP16_ONNX_PATH)
    print("Size:", round(file_size_mb, 2), "MB")

    # Verify that the external-data file was not created.
    if os.path.exists(external_data_path):
        print(
            "WARNING: External ONNX data file exists:",
            external_data_path,
        )
    else:
        print(
            "Embedded weights: YES"
        )


def build_fp16_engine():
    print()
    print("Building TensorRT FP16 engine...")

    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)

    # TensorRT 11.x uses the default strongly typed network.
    # Do NOT use:
    # trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    network = builder.create_network()

    parser = trt.OnnxParser(
        network,
        logger,
    )

    with open(
        FP16_ONNX_PATH,
        "rb",
    ) as f:
        model_data = f.read()

    print("Parsing FP16 ONNX...")

    if not parser.parse(model_data):
        print()
        print("=" * 70)
        print("TENSORRT ONNX PARSER ERRORS")
        print("=" * 70)

        for i in range(parser.num_errors):
            print()
            print(f"ERROR {i + 1}:")
            print(parser.get_error(i))

        print("=" * 70)

        raise RuntimeError(
            "TensorRT FP16 ONNX parsing failed."
        )

    print("FP16 ONNX parsing successful.")

    config = builder.create_builder_config()

    # TensorRT 11.x uses strongly typed networks.
    # Do NOT use the old:
    # config.set_flag(trt.BuilderFlag.FP16)
    #
    # The ONNX graph itself contains FP16 tensors,
    # so TensorRT builds the corresponding FP16 engine.
    config.builder_optimization_level = 3

    print()
    print("Network inputs/outputs:")

    for i in range(network.num_inputs):
        tensor = network.get_input(i)

        print(
            "INPUT:",
            tensor.name,
            "shape=",
            tensor.shape,
            "dtype=",
            tensor.dtype,
        )

    for i in range(network.num_outputs):
        tensor = network.get_output(i)

        print(
            "OUTPUT:",
            tensor.name,
            "shape=",
            tensor.shape,
            "dtype=",
            tensor.dtype,
        )

    print()
    print("Generating TensorRT FP16 engine...")

    engine_data = builder.build_serialized_network(
        network,
        config,
    )

    if engine_data is None:
        raise RuntimeError(
            "TensorRT FP16 engine build failed."
        )

    os.makedirs(
        os.path.dirname(FP16_ENGINE_PATH),
        exist_ok=True,
    )

    with open(
        FP16_ENGINE_PATH,
        "wb",
    ) as f:
        f.write(engine_data)

    engine_size_mb = (
        os.path.getsize(FP16_ENGINE_PATH)
        / (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("TensorRT FP16 engine created successfully.")
    print("=" * 70)
    print("Output:", FP16_ENGINE_PATH)
    print(
        "Size:",
        round(engine_size_mb, 2),
        "MB",
    )


def main():
    print("=" * 70)
    print("TensorRT FP16 EXPORT")
    print("=" * 70)

    print("TensorRT:", trt.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. FP16 TensorRT export requires CUDA."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print()

    export_fp16_onnx()

    build_fp16_engine()


if __name__ == "__main__":
    main()
