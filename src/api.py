import time
from threading import Lock

import cv2
import numpy as np
import torch

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from PIL import Image

from src.config import DEFAULT_THRESHOLD
from src.patchcore_pipeline import build_memory_bank
from src.preprocessing import get_test_transform
from src.feature_extractor import ResNet18FeatureExtractor
from src.production_decision import (
    make_production_decision,
)


# ============================================================
# CONFIGURATION
# ============================================================

CATEGORY = "bottle"

BATCH_SIZE = 8
SAMPLING_RATIO = 0.1

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Edge Industrial Defect Detection API",
    description=(
        "Production-oriented real-time industrial "
        "anomaly detection using ResNet18 + PatchCore."
    ),
    version="1.0.0",
)


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

feature_extractor = None
patchcore = None
transform = None

model_ready = False
startup_error = None


# ============================================================
# RUNTIME METRICS
# ============================================================

metrics_lock = Lock()

request_count = 0
successful_predictions = 0
failed_predictions = 0

latencies_ms = []


# ============================================================
# MODEL INITIALIZATION
# ============================================================

@app.on_event("startup")
def load_model():

    global feature_extractor
    global patchcore
    global transform
    global model_ready
    global startup_error

    print("=" * 70)
    print("STARTING EDGE DEFECT DETECTION API")
    print("=" * 70)

    print(
        f"Category : {CATEGORY}"
    )

    print(
        f"Device   : {DEVICE}"
    )

    print(
        f"Threshold: {DEFAULT_THRESHOLD}"
    )

    try:

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        print(
            "\nInitializing preprocessing..."
        )

        transform = get_test_transform()

        # ----------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------

        print(
            "Initializing ResNet18 feature extractor..."
        )

        feature_extractor = (
            ResNet18FeatureExtractor()
            .to(DEVICE)
        )

        feature_extractor.eval()

        # ----------------------------------------------------
        # PatchCore memory bank
        # ----------------------------------------------------

        print(
            "\nBuilding PatchCore memory bank..."
        )

        patchcore = build_memory_bank(
            category=CATEGORY,
            batch_size=BATCH_SIZE,
            sampling_ratio=SAMPLING_RATIO,
        )

        # ----------------------------------------------------
        # Model ready
        # ----------------------------------------------------

        model_ready = True
        startup_error = None

        print(
            "\nModel initialization complete."
        )

        print(
            f"Memory bank : "
            f"{patchcore.memory_bank.shape}"
        )

        print(
            f"Threshold   : "
            f"{DEFAULT_THRESHOLD}"
        )

        print("=" * 70)

    except Exception as exc:

        model_ready = False
        startup_error = str(exc)

        print(
            "\nMODEL INITIALIZATION FAILED"
        )

        print(
            f"Error: {exc}"
        )

        print("=" * 70)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    if model_ready:

        return {
            "status": "healthy",
            "model_ready": True,
            "category": CATEGORY,
            "device": str(DEVICE),
            "threshold": DEFAULT_THRESHOLD,
        }

    return {
        "status": "unhealthy",
        "model_ready": False,
        "category": CATEGORY,
        "device": str(DEVICE),
        "threshold": DEFAULT_THRESHOLD,
        "error": startup_error,
    }


# ============================================================
# MODEL INFORMATION + RUNTIME METRICS
# ============================================================

@app.get("/metrics")
def model_metrics():

    if not model_ready:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    with metrics_lock:

        count = request_count
        successful = successful_predictions
        failed = failed_predictions

        latency_values = list(
            latencies_ms
        )

    if latency_values:

        latency_array = np.asarray(
            latency_values,
            dtype=np.float32,
        )

        mean_latency = float(
            np.mean(latency_array)
        )

        p50_latency = float(
            np.percentile(
                latency_array,
                50,
            )
        )

        p95_latency = float(
            np.percentile(
                latency_array,
                95,
            )
        )

        p99_latency = float(
            np.percentile(
                latency_array,
                99,
            )
        )

        fps = float(
            1000.0 / mean_latency
        )

    else:

        mean_latency = 0.0
        p50_latency = 0.0
        p95_latency = 0.0
        p99_latency = 0.0
        fps = 0.0

    return {

        "service": {
            "status": "healthy",
            "model_ready": model_ready,
        },

        "model": {
            "category": CATEGORY,
            "device": str(DEVICE),
            "threshold": DEFAULT_THRESHOLD,
            "memory_bank_size": int(
                patchcore.memory_bank.shape[0]
            ),
            "feature_dimension": int(
                patchcore.memory_bank.shape[1]
            ),
        },

        "requests": {
            "total": count,
            "successful": successful,
            "failed": failed,
        },

        "performance": {
            "mean_latency_ms": round(
                mean_latency,
                2,
            ),
            "p50_latency_ms": round(
                p50_latency,
                2,
            ),
            "p95_latency_ms": round(
                p95_latency,
                2,
            ),
            "p99_latency_ms": round(
                p99_latency,
                2,
            ),
            "fps": round(
                fps,
                2,
            ),
        },
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    global request_count
    global successful_predictions
    global failed_predictions

    # --------------------------------------------------------
    # Request counter
    # --------------------------------------------------------

    with metrics_lock:

        request_count += 1

    # --------------------------------------------------------
    # Model readiness
    # --------------------------------------------------------

    if not model_ready:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not file.content_type.startswith(
        "image/"
    ):

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=400,
            detail=(
                "Only image files are supported."
            ),
        )

    # --------------------------------------------------------
    # Read image bytes
    # --------------------------------------------------------

    image_bytes = await file.read()

    if len(image_bytes) == 0:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    # --------------------------------------------------------
    # File-size protection
    # --------------------------------------------------------

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=413,
            detail=(
                f"Image exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            ),
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=400,
            detail="Could not decode image.",
        )

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    image_tensor = transform(
        pil_image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # GPU synchronization BEFORE timing
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            features = feature_extractor(
                image_tensor
            )

            anomaly_score, patch_distances = (
                patchcore.predict(
                    features
                )
            )

        # ----------------------------------------------------
        # GPU synchronization AFTER inference
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Production decision
        # ----------------------------------------------------

        decision = make_production_decision(
            anomaly_score=anomaly_score,
            threshold=DEFAULT_THRESHOLD,
        )

        # ----------------------------------------------------
        # Record metrics
        # ----------------------------------------------------

        with metrics_lock:

            successful_predictions += 1

            latencies_ms.append(
                latency_ms
            )

            # Prevent unlimited memory growth.
            if len(latencies_ms) > 1000:

                del latencies_ms[:-1000]

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "filename": file.filename,
            "category": CATEGORY,
            "prediction": decision.status,
            "risk_level": decision.risk_level,
            "action": decision.action,
            "anomaly_score": round(
                decision.anomaly_score,
                4,
            ),
            "threshold": round(
                decision.threshold,
                6,
            ),
            "latency_ms": round(
                latency_ms,
                2,
            ),
            "device": str(DEVICE),
        }

    except HTTPException:

        with metrics_lock:

            failed_predictions += 1

        raise

    except Exception as exc:

        with metrics_lock:

            failed_predictions += 1

        raise HTTPException(
            status_code=500,
            detail=(
                "Inference failed: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )