import time
import torch

from src.feature_extractor import ResNet18FeatureExtractor


DEVICE = "cuda"
WARMUP = 20
ITERATIONS = 100


def benchmark(model, x, name):
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    latency = (elapsed / ITERATIONS) * 1000
    fps = 1000 / latency

    print(f"\n{name}")
    print("-" * 40)
    print(f"Latency : {latency:.2f} ms")
    print(f"FPS     : {fps:.2f}")


def main():

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE
    )

    # FP32
    model_fp32 = (
        ResNet18FeatureExtractor()
        .cuda()
        .eval()
    )

    benchmark(
        model_fp32,
        x,
        "GPU FP32"
    )

    # FP16
    model_fp16 = (
        ResNet18FeatureExtractor()
        .cuda()
        .half()
        .eval()
    )

    x_fp16 = x.half()

    benchmark(
        model_fp16,
        x_fp16,
        "GPU FP16"
    )


if __name__ == "__main__":
    main()