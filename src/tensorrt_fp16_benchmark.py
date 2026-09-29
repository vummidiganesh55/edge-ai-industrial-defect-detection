import os
import time

import torch
import tensorrt as trt


FP16_ENGINE_PATH = (
    "models/resnet18_feature_extractor_fp16.engine"
)

FP32_ENGINE_PATH = (
    "models/resnet18_feature_extractor_fp32.engine"
)

WARMUP = 20
ITERATIONS = 100


# =========================================================
# LOAD TENSORRT ENGINE
# =========================================================

def load_engine(path):

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    with open(path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(
        logger
    )

    engine = runtime.deserialize_cuda_engine(
        engine_data
    )

    if engine is None:
        raise RuntimeError(
            f"Failed to load TensorRT engine: {path}"
        )

    return runtime, engine


# =========================================================
# GET INPUT / OUTPUT TENSORS
# =========================================================

def get_io_tensors(engine):

    input_name = None
    output_names = []

    for i in range(
        engine.num_io_tensors
    ):

        name = engine.get_tensor_name(i)

        mode = engine.get_tensor_mode(
            name
        )

        if mode == trt.TensorIOMode.INPUT:

            input_name = name

        else:

            output_names.append(
                name
            )

    if input_name is None:

        raise RuntimeError(
            "No input tensor found."
        )

    return (
        input_name,
        output_names
    )


# =========================================================
# RUN SINGLE TENSORRT INFERENCE
# =========================================================

def run_inference(
    engine,
    input_tensor,
):

    context = (
        engine.create_execution_context()
    )

    if context is None:

        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    (
        input_name,
        output_names
    ) = get_io_tensors(
        engine
    )

    outputs = {}

    # -----------------------------------------------------
    # Allocate output tensors
    # -----------------------------------------------------

    for name in output_names:

        shape = tuple(
            engine.get_tensor_shape(
                name
            )
        )

        dtype = engine.get_tensor_dtype(
            name
        )

        if dtype == trt.DataType.FLOAT:

            torch_dtype = (
                torch.float32
            )

        elif dtype == trt.DataType.HALF:

            torch_dtype = (
                torch.float16
            )

        else:

            raise RuntimeError(
                f"Unsupported output dtype: {dtype}"
            )

        outputs[name] = torch.empty(
            shape,
            device="cuda",
            dtype=torch_dtype
        )

    # -----------------------------------------------------
    # Set tensor addresses
    # -----------------------------------------------------

    context.set_tensor_address(
        input_name,
        input_tensor.data_ptr()
    )

    for name in output_names:

        context.set_tensor_address(
            name,
            outputs[name].data_ptr()
        )

    # -----------------------------------------------------
    # Execute inference
    # -----------------------------------------------------

    stream = torch.cuda.Stream()

    success = (
        context.execute_async_v3(
            stream.cuda_stream
        )
    )

    if not success:

        raise RuntimeError(
            "TensorRT inference failed."
        )

    stream.synchronize()

    del stream
    del context

    return outputs


# =========================================================
# FP16 BENCHMARK
# =========================================================

def benchmark_fp16(
    engine,
    input_tensor,
):

    context = (
        engine.create_execution_context()
    )

    if context is None:

        raise RuntimeError(
            "Failed to create TensorRT execution context."
        )

    (
        input_name,
        output_names
    ) = get_io_tensors(
        engine
    )

    outputs = {}

    # -----------------------------------------------------
    # Allocate FP16 outputs
    # -----------------------------------------------------

    for name in output_names:

        shape = tuple(
            engine.get_tensor_shape(
                name
            )
        )

        outputs[name] = torch.empty(
            shape,
            device="cuda",
            dtype=torch.float16
        )

    # -----------------------------------------------------
    # CUDA stream
    # -----------------------------------------------------

    stream = torch.cuda.Stream()

    # -----------------------------------------------------
    # Tensor addresses
    # -----------------------------------------------------

    context.set_tensor_address(
        input_name,
        input_tensor.data_ptr()
    )

    for name in output_names:

        context.set_tensor_address(
            name,
            outputs[name].data_ptr()
        )

    # -----------------------------------------------------
    # Warmup
    # -----------------------------------------------------

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
                "TensorRT FP16 inference "
                "failed during warmup."
            )

    stream.synchronize()

    # -----------------------------------------------------
    # Benchmark
    # -----------------------------------------------------

    print(
        "Running FP16 benchmark..."
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
                "TensorRT FP16 inference failed."
            )

    stream.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    latency = (
        elapsed
        / ITERATIONS
        * 1000
    )

    fps = (
        1000
        / latency
    )

    return (
        latency,
        fps,
        outputs
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # SYSTEM INFORMATION
    # =====================================================

    print(
        "TensorRT:",
        trt.__version__
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "FP16 Engine:",
        FP16_ENGINE_PATH
    )

    print(
        "FP16 Engine size:",
        round(
            os.path.getsize(
                FP16_ENGINE_PATH
            )
            / (1024 * 1024),
            2
        ),
        "MB"
    )

    # =====================================================
    # LOAD FP16 ENGINE
    # =====================================================

    (
        fp16_runtime,
        fp16_engine
    ) = load_engine(
        FP16_ENGINE_PATH
    )

    print(
        "\nTensorRT FP16 engine "
        "loaded successfully."
    )

    # =====================================================
    # DISPLAY FP16 TENSORS
    # =====================================================

    print(
        "\nTensors:"
    )

    for i in range(
        fp16_engine.num_io_tensors
    ):

        name = (
            fp16_engine.get_tensor_name(i)
        )

        mode = (
            fp16_engine.get_tensor_mode(
                name
            )
        )

        shape = (
            fp16_engine.get_tensor_shape(
                name
            )
        )

        dtype = (
            fp16_engine.get_tensor_dtype(
                name
            )
        )

        print(
            f"  {name} | "
            f"{mode} | "
            f"shape={shape} | "
            f"dtype={dtype}"
        )

    # =====================================================
    # CREATE FP16 INPUT
    # =====================================================

    x_fp16 = torch.randn(
        1,
        3,
        224,
        224,
        device="cuda",
        dtype=torch.float16
    )

    # =====================================================
    # RUN FP16 BENCHMARK
    # =====================================================

    (
        latency,
        fps,
        fp16_outputs
    ) = benchmark_fp16(
        fp16_engine,
        x_fp16
    )

    # =====================================================
    # FP16 RESULTS
    # =====================================================

    print()

    print(
        "=" * 50
    )

    print(
        "TensorRT FP16"
    )

    print(
        "=" * 50
    )

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

    # =====================================================
    # OUTPUT SHAPES
    # =====================================================

    print(
        "\nOutput shapes:"
    )

    for (
        name,
        tensor
    ) in fp16_outputs.items():

        print(
            f"  {name}: "
            f"{tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )

    # =====================================================
    # FP32 vs FP16 CONSISTENCY
    # =====================================================

    print()

    print(
        "=" * 50
    )

    print(
        "FP32 vs FP16 FEATURE CONSISTENCY"
    )

    print(
        "=" * 50
    )

    # -----------------------------------------------------
    # Load FP32 engine
    # -----------------------------------------------------

    (
        fp32_runtime,
        fp32_engine
    ) = load_engine(
        FP32_ENGINE_PATH
    )

    print(
        "TensorRT FP32 engine "
        "loaded successfully."
    )

    # -----------------------------------------------------
    # Convert same input to FP32
    # -----------------------------------------------------

    x_fp32 = (
        x_fp16.float()
    )

    # -----------------------------------------------------
    # Run FP32 inference
    # -----------------------------------------------------

    fp32_outputs = run_inference(
        fp32_engine,
        x_fp32
    )

    # -----------------------------------------------------
    # Numerical comparison
    # -----------------------------------------------------

    print(
        "\nFP32 vs FP16 numerical comparison:"
    )

    all_passed = True

    for name in fp16_outputs:

        # Convert FP16 output to FP32
        fp16_output = (
            fp16_outputs[name]
            .float()
        )

        fp32_output = (
            fp32_outputs[name]
        )

        # -------------------------------------------------
        # Absolute difference
        # -------------------------------------------------

        difference = torch.abs(
            fp32_output
            - fp16_output
        )

        max_abs_error = (
            torch.max(
                difference
            ).item()
        )

        mean_abs_error = (
            torch.mean(
                difference
            ).item()
        )

        # -------------------------------------------------
        # Relative error
        # -------------------------------------------------

        denominator = (
            torch.abs(
                fp32_output
            )
            + 1e-8
        )

        relative_error = (
            torch.mean(
                difference
                / denominator
            ).item()
        )

        # -------------------------------------------------
        # Cosine similarity
        # -------------------------------------------------

        fp32_flat = (
            fp32_output
            .reshape(-1)
        )

        fp16_flat = (
            fp16_output
            .reshape(-1)
        )

        cosine_similarity = (
            torch.nn.functional
            .cosine_similarity(
                fp32_flat.unsqueeze(0),
                fp16_flat.unsqueeze(0),
                dim=1
            )
            .item()
        )

        # -------------------------------------------------
        # Print metrics
        # -------------------------------------------------

        print()

        print(
            f"{name}"
        )

        print(
            f"  Max absolute error : "
            f"{max_abs_error:.8f}"
        )

        print(
            f"  Mean absolute error: "
            f"{mean_abs_error:.8f}"
        )

        print(
            f"  Mean relative error: "
            f"{relative_error:.8f}"
        )

        print(
            f"  Cosine similarity  : "
            f"{cosine_similarity:.8f}"
        )

        # -------------------------------------------------
        # Validation criterion
        #
        # Cosine similarity is the primary metric because
        # PatchCore uses feature representations.
        # -------------------------------------------------

        if cosine_similarity < 0.999:

            all_passed = False

    # =====================================================
    # VALIDATION RESULT
    # =====================================================

    print()

    if all_passed:

        print(
            "FP32 vs FP16 consistency: PASSED"
        )

    else:

        print(
            "FP32 vs FP16 consistency: "
            "REVIEW REQUIRED"
        )

    # =====================================================
    # CLEANUP
    # =====================================================

    del fp16_outputs
    del fp32_outputs

    del x_fp16
    del x_fp32

    del fp16_engine
    del fp32_engine

    del fp16_runtime
    del fp32_runtime

    torch.cuda.empty_cache()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
