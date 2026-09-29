from pathlib import Path
import io
import json

import pandas as pd
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

API_URL = "http://localhost:8000"

PRODUCTION_THRESHOLD = 17.364517


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Edge Industrial Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
    }

    .subtitle {
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_json(path: Path):
    """Safely load JSON."""

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as exc:
        st.error(
            f"Could not read {path.name}: {exc}"
        )
        return None


def load_csv(path: Path):
    """Safely load CSV."""

    try:
        return pd.read_csv(path)

    except Exception as exc:
        st.error(
            f"Could not read {path.name}: {exc}"
        )
        return None


def find_file(filename: str):
    """Search for a file inside outputs."""

    matches = list(
        OUTPUTS_DIR.rglob(filename)
    )

    if matches:
        return matches[0]

    return None


def api_health():
    """Check FastAPI health."""

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except Exception:
        return None


def predict_image(uploaded_file):
    """Send image to FastAPI production endpoint."""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    response = requests.post(
        f"{API_URL}/predict",
        files=files,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


def display_prediction(result):
    """Display production inference result."""

    prediction = result.get(
        "prediction",
        "UNKNOWN",
    )

    risk = result.get(
        "risk_level",
        "UNKNOWN",
    )

    action = result.get(
        "action",
        "UNKNOWN",
    )

    score = result.get(
        "anomaly_score",
        0.0,
    )

    threshold = result.get(
        "threshold",
        PRODUCTION_THRESHOLD,
    )

    latency = result.get(
        "latency_ms",
        0.0,
    )

    device = result.get(
        "device",
        "unknown",
    )

    st.markdown(
        "### 🎯 Production Result"
    )

    if prediction == "DEFECT":

        st.error(
            f"""
## 🔴 DEFECT

**Risk Level:** {risk}

**Action:** {action}
"""
        )

    elif prediction == "NORMAL":

        st.success(
            f"""
## 🟢 NORMAL

**Risk Level:** {risk}

**Action:** {action}
"""
        )

    else:

        st.warning(
            f"""
## ⚠️ {prediction}

**Risk Level:** {risk}

**Action:** {action}
"""
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Anomaly Score",
            f"{float(score):.4f}",
        )

    with col2:

        st.metric(
            "Threshold",
            f"{float(threshold):.4f}",
        )

    with col3:

        st.metric(
            "Latency",
            f"{float(latency):.2f} ms",
        )

    with col4:

        st.metric(
            "Device",
            str(device).upper(),
        )

    st.markdown(
        "### 📈 Anomaly Score"
    )

    try:

        score_float = float(score)
        threshold_float = float(threshold)

        maximum = max(
            score_float,
            threshold_float,
            1.0,
        )

        st.progress(
            min(
                score_float / maximum,
                1.0,
            )
        )

        st.caption(
            f"Score = {score_float:.4f} | "
            f"Production threshold = "
            f"{threshold_float:.4f}"
        )

    except Exception:

        pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔍 Edge AI")

st.sidebar.markdown(
    """
**Industrial Defect Detection**

PatchCore-based anomaly detection
with production decisioning and
edge deployment.
"""
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔍 Live Detection",
        "🗺️ Anomaly Localization",
        "📊 Model Evaluation",
        "🔬 Failure Analysis",
        "🧪 Robustness",
        "⚡ Edge Benchmarks",
        "📦 Quantization",
        "📡 Monitoring",
        "🚀 API / Docker",
        "📋 System Information",
    ],
)

st.sidebar.divider()

health = api_health()

if health:

    st.sidebar.success(
        "API: ONLINE"
    )

    st.sidebar.caption(
        f"Device: {health.get('device', 'unknown')}"
    )

    st.sidebar.caption(
        f"Threshold: "
        f"{health.get('threshold', PRODUCTION_THRESHOLD)}"
    )

else:

    st.sidebar.warning(
        "API: OFFLINE"
    )

    st.sidebar.caption(
        "Start FastAPI for live detection."
    )


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '🔍 Edge-Deployed Real-Time Industrial '
        'Defect Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Production-oriented visual anomaly detection '
        'using PatchCore and edge inference optimization.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Image AUROC",
            "0.9968",
        )

    with col2:
        st.metric(
            "Accuracy",
            "97.59%",
        )

    with col3:
        st.metric(
            "F1 Score",
            "98.39%",
        )

    with col4:
        st.metric(
            "FPR",
            "0%",
        )

    st.markdown(
        "## 🎯 Project Objective"
    )

    st.write(
        """
Detect industrial product defects from visual inspection
images using PatchCore-based anomaly detection,
production thresholding, anomaly localization,
robustness testing, monitoring, and edge deployment
optimization.
"""
    )

    st.markdown(
        "## 🏗️ System Architecture"
    )

    st.code(
        """
Image
  │
  ▼
Preprocessing
  │
  ▼
ResNet18 Feature Extraction
  │
  ▼
PatchCore Memory Bank
  │
  ▼
Anomaly Score
  │
  ▼
Production Decision
  │
  ├── NORMAL
  │
  └── DEFECT
       │
       ├── LOW
       ├── MEDIUM
       └── HIGH
""",
        language="text",
    )

    st.markdown(
        "## 🧠 Technology Stack"
    )

    technologies = [
        "Python",
        "PyTorch",
        "ResNet18",
        "PatchCore",
        "OpenCV",
        "FastAPI",
        "Streamlit",
        "Docker",
        "ONNX Runtime",
        "OpenVINO",
        "TensorRT",
        "CUDA",
    ]

    st.write(
        " • ".join(technologies)
    )

    st.markdown(
        "## 📌 Production Configuration"
    )

    st.info(
        f"Production anomaly threshold: "
        f"**{PRODUCTION_THRESHOLD}**"
    )


# ============================================================
# 2. LIVE DETECTION
# ============================================================

elif page == "🔍 Live Detection":

    st.title(
        "🔍 Live Industrial Defect Detection"
    )

    st.write(
        """
Upload an industrial image and process it through
the production FastAPI + PatchCore inference pipeline.
"""
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        help=(
            "Upload an industrial inspection image."
        ),
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🖼️ Input Image"
                )

                st.image(
                    image,
                    caption=uploaded_file.name,
                    width="stretch",
                )

                st.caption(
                    f"Image size: "
                    f"{image.width} × {image.height}"
                )

            with col2:

                st.markdown(
                    "### ⚙️ Processing"
                )

                st.write(
                    """
The image will be sent to the production
inference API.

**Pipeline**

Upload → Preprocessing → ResNet18 →
PatchCore → Anomaly Score →
Production Decision
"""
                )

                process = st.button(
                    "🚀 Process Image",
                    type="primary",
                    width="stretch",
                )

            if process:

                if health is None:

                    st.error(
                        """
FastAPI is not running.

Start it with:

`uvicorn src.api:app --host 0.0.0.0 --port 8000`
"""
                    )

                else:

                    with st.spinner(
                        "Running PatchCore inference..."
                    ):

                        try:

                            result = predict_image(
                                uploaded_file
                            )

                            st.session_state[
                                "last_prediction"
                            ] = result

                            st.session_state[
                                "last_image"
                            ] = image_bytes

                            st.success(
                                "Image processed successfully."
                            )

                        except requests.exceptions.RequestException as exc:

                            st.error(
                                f"API request failed: {exc}"
                            )

                        except Exception as exc:

                            st.error(
                                f"Prediction failed: {exc}"
                            )

        except Exception as exc:

            st.error(
                f"Could not open uploaded image: {exc}"
            )

    if "last_prediction" in st.session_state:

        st.divider()

        result = st.session_state[
            "last_prediction"
        ]

        display_prediction(result)

        st.markdown(
            "### 📄 Prediction Details"
        )

        result_rows = []

        for key, value in result.items():

            result_rows.append(
                {
                    "Property": str(key),
                    "Value": str(value),
                }
            )

        result_df = pd.DataFrame(
            result_rows
        )

        st.dataframe(
            result_df,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# 3. ANOMALY LOCALIZATION
# ============================================================

elif page == "🗺️ Anomaly Localization":

    st.title(
        "🗺️ Anomaly Localization"
    )

    st.write(
        """
Pre-generated PatchCore anomaly maps and
localization results from the evaluation pipeline.
"""
    )

    localization_dir = (
        OUTPUTS_DIR / "localization"
    )

    if not localization_dir.exists():

        st.warning(
            "Localization output directory was not found."
        )

    else:

        localization_files = sorted(
            [
                p
                for p in localization_dir.iterdir()
                if p.is_file()
            ]
        )

        if not localization_files:

            st.info(
                "No localization files found."
            )

        else:

            selected = st.selectbox(
                "Select localization result",
                localization_files,
                format_func=lambda x: x.name,
            )

            st.markdown(
                f"**Selected:** `{selected.name}`"
            )

            try:

                suffix = selected.suffix.lower()

                if suffix in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                ]:

                    st.image(
                        Image.open(selected),
                        width="stretch",
                    )

                elif suffix == ".json":

                    data = load_json(
                        selected
                    )

                    if data is not None:
                        st.json(data)

                else:

                    st.code(
                        selected.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        )
                    )

            except Exception as exc:

                st.error(
                    f"Could not display file: {exc}"
                )


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

elif page == "📊 Model Evaluation":

    st.title(
        "📊 Model Evaluation"
    )

    st.write(
        "Authoritative image-level PatchCore evaluation."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "AUROC",
            "0.9968",
        )

    with col2:
        st.metric(
            "Accuracy",
            "97.59%",
        )

    with col3:
        st.metric(
            "Precision",
            "100%",
        )

    with col4:
        st.metric(
            "Recall",
            "96.83%",
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "F1",
            "98.39%",
        )

    with col6:
        st.metric(
            "FPR",
            "0%",
        )

    with col7:
        st.metric(
            "TN",
            "20",
        )

    with col8:
        st.metric(
            "FN",
            "2",
        )

    st.divider()

    st.markdown(
        "### 🎯 Production Threshold"
    )

    st.info(
        f"Locked production threshold: "
        f"**{PRODUCTION_THRESHOLD}**"
    )

    st.markdown(
        "### Confusion Matrix"
    )

    cm = pd.DataFrame(
        [
            [20, 0],
            [2, 61],
        ],
        index=[
            "Actual Normal",
            "Actual Defect",
        ],
        columns=[
            "Predicted Normal",
            "Predicted Defect",
        ],
    )

    st.dataframe(
        cm,
        width="stretch",
    )

    st.markdown(
        "### 📈 Evaluation Output"
    )

    evaluation_file = find_file(
        "patchcore_evaluation.json"
    )

    if evaluation_file:

        data = load_json(
            evaluation_file
        )

        if data is not None:
            st.json(data)

    else:

        st.info(
            "patchcore_evaluation.json not found."
        )


# ============================================================
# 5. FAILURE ANALYSIS
# ============================================================

elif page == "🔬 Failure Analysis":

    st.title(
        "🔬 Failure Analysis"
    )

    st.markdown(
        "### ❌ False Positives"
    )

    st.success(
        "No false positives at the production threshold."
    )

    st.markdown(
        "### ⚠️ False Negatives"
    )

    st.warning(
        """
2 false negatives were observed.

Both belonged to the **contamination** category.
"""
    )

    st.markdown(
        "### 🧪 Defect-Type Analysis"
    )

    failure_file = (
        OUTPUTS_DIR
        / "evaluation"
        / "failure_analysis.txt"
    )

    if failure_file.exists():

        content = failure_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        st.code(content)

    else:

        st.info(
            "failure_analysis.txt not found."
        )


# ============================================================
# 6. ROBUSTNESS
# ============================================================

elif page == "🧪 Robustness":

    st.title(
        "🧪 Robustness Testing"
    )

    st.write(
        "Performance under controlled image perturbations."
    )

    robustness_dir = (
        OUTPUTS_DIR / "robustness"
    )

    if robustness_dir.exists():

        robustness_files = sorted(
            [
                p
                for p in robustness_dir.iterdir()
                if p.is_file()
            ]
        )

        for file in robustness_files:

            st.markdown(
                f"### `{file.name}`"
            )

            suffix = file.suffix.lower()

            try:

                if suffix == ".json":

                    data = load_json(file)

                    if isinstance(data, dict):
                        st.json(data)

                    elif isinstance(data, list):

                        df = pd.DataFrame(data)

                        st.dataframe(
                            df,
                            width="stretch",
                        )

                elif suffix == ".csv":

                    df = load_csv(file)

                    if df is not None:

                        st.dataframe(
                            df,
                            width="stretch",
                        )

                elif suffix in [
                    ".txt",
                    ".log",
                ]:

                    st.code(
                        file.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        )
                    )

            except Exception as exc:

                st.error(
                    f"Could not display {file.name}: {exc}"
                )

    else:

        st.info(
            "Robustness output directory not found."
        )

    st.markdown(
        "### ⚠️ Known Limitation"
    )

    st.warning(
        """
Gaussian noise produced significant performance
degradation during robustness testing.

This is documented as a known limitation rather
than changing the locked production threshold.
"""
    )


# ============================================================
# 7. EDGE BENCHMARKS
# ============================================================

elif page == "⚡ Edge Benchmarks":

    st.title(
        "⚡ Edge Inference Benchmarks"
    )

    st.caption(
        """
Important: TensorRT, ONNX Runtime and OpenVINO
numbers are feature-extractor benchmarks.

The CPU PatchCore benchmark represents full
PatchCore inference.
"""
    )

    benchmark_df = pd.DataFrame(
        [
            {
                "Runtime": "Full PatchCore CPU",
                "Latency (ms)": 96.53,
                "FPS": 10.36,
                "Scope": "Full PatchCore",
            },
            {
                "Runtime": "ONNX Runtime CPU",
                "Latency (ms)": 9.20,
                "FPS": 108.74,
                "Scope": "Feature extractor",
            },
            {
                "Runtime": "OpenVINO CPU",
                "Latency (ms)": 15.67,
                "FPS": 63.84,
                "Scope": "Feature extractor",
            },
            {
                "Runtime": "TensorRT FP32",
                "Latency (ms)": 1.43,
                "FPS": 700.81,
                "Scope": "Feature extractor",
            },
            {
                "Runtime": "TensorRT FP16",
                "Latency (ms)": 0.91,
                "FPS": 1100.0,
                "Scope": "Feature extractor",
            },
        ]
    )

    st.dataframe(
        benchmark_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### 🚀 TensorRT Comparison"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "TensorRT FP32",
            "1.43 ms",
            "700.81 FPS",
        )

    with col2:

        st.metric(
            "TensorRT FP16",
            "0.91 ms",
            "~1100 FPS",
        )

    st.info(
        """
FP16 reduces engine size from 17.73 MB
to 8.01 MB and provides approximately
1.57× lower latency.
"""
    )

    cpu_file = (
        OUTPUTS_DIR
        / "benchmark"
        / "cpu_baseline_results.csv"
    )

    if cpu_file.exists():

        st.markdown(
            "### 🖥️ CPU Benchmark Output"
        )

        cpu_df = load_csv(
            cpu_file
        )

        if cpu_df is not None:

            st.dataframe(
                cpu_df,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# 8. QUANTIZATION
# ============================================================

elif page == "📦 Quantization":

    st.title(
        "📦 Quantization Evaluation"
    )

    quant_df = pd.DataFrame(
        [
            {
                "Model": "FP32 ONNX",
                "Size": "10.61 MB",
                "Latency": "9.58 ms",
                "FPS": "104.33",
                "Status": "Selected",
            },
            {
                "Model": "Dynamic INT8 ONNX",
                "Size": "2.69 MB",
                "Latency": "203.40 ms",
                "FPS": "4.92",
                "Status": "Not selected",
            },
        ]
    )

    st.dataframe(
        quant_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### 📉 Compression"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "INT8 Size Reduction",
            "74.66%",
        )

    with col2:

        st.metric(
            "INT8 Compression",
            "3.95× smaller",
        )

    st.markdown(
        "### ⚠️ Deployment Assessment"
    )

    st.warning(
        """
Dynamic INT8 quantization was evaluated but
not selected for deployment in this
runtime/configuration.

The model became substantially smaller,
but latency increased significantly and
feature consistency required review.
"""
    )


# ============================================================
# 9. MONITORING
# ============================================================

elif page == "📡 Monitoring":

    st.title(
        "📡 Production Monitoring"
    )

    monitoring_file = (
        OUTPUTS_DIR
        / "monitoring"
        / "monitoring_report.json"
    )

    if monitoring_file.exists():

        data = load_json(
            monitoring_file
        )

        if isinstance(data, dict):

            st.json(data)

    else:

        st.warning(
            "Monitoring report not found."
        )

    st.markdown(
        "### Monitoring Snapshot"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Anomaly PSI",
            "0.1407",
        )

    with col2:

        st.metric(
            "Brightness PSI",
            "0.1648",
        )

    with col3:

        st.metric(
            "Contrast PSI",
            "0.0549",
        )

    st.warning(
        """
Overall monitoring status: WARNING

Moderate drift was detected in the
demonstration monitoring data.
"""
    )


# ============================================================
# 10. API / DOCKER
# ============================================================

elif page == "🚀 API / Docker":

    st.title(
        "🚀 API / Docker"
    )

    health = api_health()

    if health:

        st.success(
            "FastAPI is ONLINE"
        )

        st.json(health)

    else:

        st.error(
            "FastAPI is OFFLINE"
        )

        st.code(
            """
uvicorn src.api:app --host 0.0.0.0 --port 8000
""",
            language="powershell",
        )

    st.markdown(
        "### 🐳 Docker Deployment"
    )

    docker_metrics = {
        "Container": "edge-defect-container",
        "Image": "edge-defect-api:latest",
        "API": "FastAPI",
        "Port": "8000",
        "Device": "CPU",
        "Health": "Healthy",
        "Production Threshold": PRODUCTION_THRESHOLD,
    }

    docker_rows = [
        {
            "Component": key,
            "Value": str(value),
        }
        for key, value in docker_metrics.items()
    ]

    docker_df = pd.DataFrame(
        docker_rows
    )

    st.dataframe(
        docker_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### ✅ Docker Validation"
    )

    st.success(
        """
Docker image build and container API
validation completed successfully.
"""
    )


# ============================================================
# 11. SYSTEM INFORMATION
# ============================================================

elif page == "📋 System Information":

    st.title(
        "📋 System Information"
    )

    information = {
        "Dataset": "MVTec AD — Bottle",
        "Training Images": 209,
        "Test Images": 83,
        "Normal Test Images": 20,
        "Defect Test Images": 63,
        "Model": "PatchCore",
        "Feature Extractor": "ResNet18",
        "Memory Bank": "16,385 × 384",
        "Production Threshold": PRODUCTION_THRESHOLD,
        "GPU": "NVIDIA GeForce GTX 1650",
        "TensorRT": "11.3.0.99",
        "PyTorch": "2.14.0+cu130",
        "Deployment": "FastAPI + Docker",
        "Dashboard": "Streamlit",
    }

    # IMPORTANT:
    # Convert every value to string so Arrow receives
    # a consistent dataframe column type.

    info_rows = [
        {
            "Component": str(key),
            "Value": str(value),
        }
        for key, value in information.items()
    ]

    info_df = pd.DataFrame(
        info_rows
    )

    st.dataframe(
        info_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "### ⚠️ Known Limitations"
    )

    st.write(
        """
• Contamination contains the observed false negatives.

• Gaussian noise causes significant robustness degradation.

• Current INT8 dynamic quantization was not selected
  for deployment.

• TensorRT, ONNX Runtime and OpenVINO benchmark
  numbers are feature-extractor measurements.

• The Docker deployment uses CPU PyTorch in the
  current image.

• The production threshold is locked at 17.364517.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Edge Industrial Defect Detection"
)

st.sidebar.caption(
    "PatchCore • FastAPI • Docker • Streamlit"
)