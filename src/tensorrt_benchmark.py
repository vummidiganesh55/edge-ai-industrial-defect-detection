import os
import time

import numpy as np
import torch
import tensorrt as trt


ENGINE_PATH = "models/resnet18_feature_extractor_fp32.engine"

WARMUP = 20
ITERATIONS = 100


def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)

    with open(path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            "Failed to load TensorRT engine."
        )

    return runtime, engine


def main():

    # ---------------------------------------------------------
    # System information
    # ---------------------------------------------------------

    print("TensorRT:", trt.__version__)
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Engine:",
        ENGINE_PATH
    )

    print(
        "Engine size:",
        round(
            os.path.getsize(ENGINE_PATH)
            / (1024 * 1024),
            2
        ),
        "MB"
    )

    # ---------------------------------------------------------
    # Load TensorRT engine
    # ---------------------------------------------------------

    runtime, engine = load_engine(
        ENGINE_PATH
    )

    print(
        "\nTensorRT engine loaded successfully."
    )

    # ---------------------------------------------------------
    # Display tensors
    # ---------------------------------------------------------

    print("\nTensors:")

    for i in range(
        engine.num_io_tensors
    ):

        name = engine.get_tensor_name(i)

        mode = engine.get_tensor_mode(
            name
        )

        shape = engine.get_tensor_shape(
            name
        )

        dtype = engine.get_tensor_dtype(
            name
        )

        print(
            f"  {name} | "
            f"{mode} | "
            f"shape={shape} | "
            f"dtype={dtype}"
        )

    # ---------------------------------------------------------
    # Create execution context
    # ---------------------------------------------------------

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    # ---------------------------------------------------------
    # Find input/output tensors
    # ---------------------------------------------------------

    input_name = None
    output_names = []

    for i in range(
        engine.num_io_tensors
    ):

        name = engine.get_tensor_name(i)

        if (
            engine.get_tensor_mode(name)
            == trt.TensorIOMode.INPUT
        ):
            input_name = name

        else:
            output_names.append(name)

    if input_name is None:
        raise RuntimeError(
            "No input tensor found."
        )

    # ---------------------------------------------------------
    # Input tensor
    # ---------------------------------------------------------

    x = torch.randn(
        1,
        3,
        224,
        224,
        device="cuda",
        dtype=torch.float32
    )

    # ---------------------------------------------------------
    # Output tensors
    # ---------------------------------------------------------

    outputs = {}

    for name in output_names:

        shape = tuple(
            engine.get_tensor_shape(name)
        )

        outputs[name] = torch.empty(
            shape,
            device="cuda",
            dtype=torch.float32
        )

    # ---------------------------------------------------------
    # Dedicated CUDA stream
    # ---------------------------------------------------------

    stream = torch.cuda.Stream()

    # ---------------------------------------------------------
    # TensorRT tensor addresses
    # ---------------------------------------------------------

    context.set_tensor_address(
        input_name,
        x.data_ptr()
    )

    for name in output_names:

        context.set_tensor_address(
            name,
            outputs[name].data_ptr()
        )

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    print(
        "\nRunning warmup..."
    )

    for _ in range(WARMUP):

        success = (
            context.execute_async_v3(
                stream.cuda_stream
            )
        )

        if not success:
            raise RuntimeError(
                "TensorRT inference failed during warmup."
            )

    stream.synchronize()

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    print(
        "Running benchmark..."
    )

    start = time.perf_counter()

    for _ in range(ITERATIONS):

        success = (
            context.execute_async_v3(
                stream.cuda_stream
            )
        )

        if not success:
            raise RuntimeError(
                "TensorRT inference failed."
            )

    # Wait for GPU completion
    stream.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    latency = (
        elapsed
        / ITERATIONS
        * 1000
    )

    fps = (
        1000
        / latency
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 50)
    print("TensorRT FP32")
    print("=" * 50)

    print(
        f"Latency : {latency:.2f} ms"
    )

    print(
        f"FPS     : {fps:.2f}"
    )

    print(
        f"Warmup  : {WARMUP}"
    )

    print(
        f"Iterations: {ITERATIONS}"
    )

    # ---------------------------------------------------------
    # Output shapes
    # ---------------------------------------------------------

    print(
        "\nOutput shapes:"
    )

    for name, tensor in outputs.items():

        print(
            f"  {name}: "
            f"{tuple(tensor.shape)}"
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    del outputs
    del x
    del context
    del engine
    del runtime

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
