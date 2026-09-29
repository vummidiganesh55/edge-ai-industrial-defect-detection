import os
import tensorrt as trt


ONNX_PATH = "models/resnet18_feature_extractor.onnx"
OUTPUT_PATH = "models/resnet18_feature_extractor_fp32.engine"


def build_engine():

    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)

    network = builder.create_network()

    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as f:
        model_data = f.read()

    if not parser.parse(model_data):
        print("ONNX parsing failed.")

        for i in range(parser.num_errors):
            print(parser.get_error(i))

        return

    config = builder.create_builder_config()

    config.builder_optimization_level = 3

    print("Building TensorRT FP32 engine...")

    engine = builder.build_serialized_network(
        network,
        config
    )

    if engine is None:
        print("Engine build failed.")
        return

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(OUTPUT_PATH, "wb") as f:
        f.write(engine)

    print()
    print("TensorRT engine created successfully.")
    print("Output:", OUTPUT_PATH)
    print(
        "Size:",
        round(os.path.getsize(OUTPUT_PATH) / (1024 * 1024), 2),
        "MB"
    )


if __name__ == "__main__":
    build_engine()