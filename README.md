Absolutely. Based on your completed **Edge-Deployed Real-Time Industrial Defect Detection** project and the results you shared, here is a **GitHub-ready README** following your exact structure.

You can copy-paste this directly into `README.md`.

> **Screenshot setup:** create a folder such as `docs/results/` in your repository and place the uploaded result screenshots there using these filenames:
> `cpu_baseline.png`, `failure_analysis.png`, `fp32_vs_fp16.png`, `fp32_vs_int8.png`, `monitoring_drift.png`, `ONNX_runtime.png`, `OPENVIO_result.png`, `patchcore_metrics.png`, `pixel_evaluation.png`, `pro_eval.png`, `py_test.png`.

---

````markdown
# 🚀 Edge-Deployed Real-Time Industrial Defect Detection

## ⭐ Badges

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-green)
![ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime-purple)
![OpenVINO](https://img.shields.io/badge/Optimization-OpenVINO-red)
![TensorRT](https://img.shields.io/badge/Optimization-TensorRT-76B900)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue)
![Testing](https://img.shields.io/badge/Tests-4%20Passed-success)

---

# 📌 Description

An end-to-end **edge AI industrial defect detection system** designed for real-time visual inspection in manufacturing environments.

The system uses **PatchCore anomaly detection** to identify previously unseen defects without requiring defect-specific training data.

The project focuses on building a production-oriented computer vision pipeline covering:

- Industrial anomaly detection
- Unseen defect detection
- Image-level classification
- Pixel-level localization
- Threshold calibration
- Failure analysis
- CPU/GPU benchmarking
- ONNX deployment
- OpenVINO optimization
- TensorRT FP16 acceleration
- INT8 quantization analysis
- Feature consistency validation
- Production monitoring
- Data drift detection
- Automated testing
- Docker-ready deployment

The system was evaluated using the **MVTec AD dataset**, with the `bottle` category used for the primary anomaly detection experiments.

---

# 🎯 Problem Statement

Traditional industrial inspection systems often depend on supervised defect classifiers that require large numbers of labeled defect images.

This creates several challenges:

- New and unseen defects may not exist in the training dataset.
- Manufacturing environments can change over time.
- Lighting and camera conditions can shift.
- Product appearance can vary.
- False negatives can allow defective products to pass inspection.
- False positives can increase unnecessary manual inspections.
- Edge devices have limited CPU, GPU, memory, and power resources.
- Production systems require low latency and high throughput.
- Models must be monitored after deployment.
- Data drift can reduce model reliability over time.

The goal of this project is to build an **anomaly-based industrial inspection system that can detect abnormal products while remaining suitable for edge deployment and production monitoring.**

---

# 💡 Solution

The system follows an anomaly detection approach.

Instead of training a classifier using every possible defect type, the model learns the representation of **normal products** and identifies deviations from the learned normal feature distribution.

The main pipeline is:

```text
Industrial Image
       │
       ▼
Preprocessing
       │
       ▼
Feature Extraction
       │
       ▼
PatchCore Memory Bank
       │
       ▼
Nearest Neighbor Anomaly Scoring
       │
       ▼
Anomaly Score
       │
       ▼
Production Threshold
       │
       ├───────────────┐
       ▼               ▼
    NORMAL          DEFECT
       │               │
       └───────┬───────┘
               ▼
      Monitoring & Logging
               │
               ▼
       Drift Detection
````

PatchCore provides both:

* **Image-level anomaly detection**
* **Pixel-level anomaly localization**

The project additionally evaluates whether the feature extractor can be optimized for edge inference using:

* ONNX Runtime
* OpenVINO
* TensorRT FP16
* TensorRT/ONNX INT8 analysis

---

# 🚀 Features

## Core AI

* PatchCore anomaly detection
* Unsupervised / semi-supervised anomaly learning
* Normal-image memory bank
* Nearest-neighbor anomaly scoring
* Image-level defect classification
* Pixel-level anomaly localization
* Production threshold calibration
* Youden threshold analysis
* Failure analysis

## Computer Vision

* Image preprocessing
* Image normalization
* Feature extraction
* Patch-level feature representation
* Anomaly heatmap generation
* Defect localization
* Industrial image analysis

## Edge AI Optimization

* PyTorch inference
* ONNX export
* ONNX Runtime inference
* OpenVINO CPU inference
* TensorRT FP16 inference
* INT8 model-size analysis
* FP32 vs FP16 consistency testing
* FP32 vs INT8 consistency testing
* Model compression analysis

## Production Engineering

* CPU benchmarking
* GPU benchmarking
* Latency measurement
* P50/P95/P99 latency
* FPS measurement
* Throughput measurement
* Memory usage monitoring
* Failure analysis
* Automated testing
* Monitoring dashboard/report generation
* Data drift detection
* Prediction monitoring

## Deployment

* Docker-ready architecture
* FastAPI-compatible inference architecture
* Streamlit-ready demonstration layer
* Edge deployment support

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Industrial Camera  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Image Preprocessing  │
                         │ Resize / Normalize   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Feature Extractor       │
                    │ ResNet-based CNN Backbone    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      PatchCore Engine         │
                    │                               │
                    │ Patch Features                │
                    │ Memory Bank                   │
                    │ Nearest Neighbor Search       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Anomaly Score      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Production Threshold │
                         │ 17.364517            │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                     NORMAL                  DEFECT
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Monitoring + Drift Detection │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Edge Deployment / API / UI    │
                    │ ONNX / OpenVINO / TensorRT    │
                    └───────────────────────────────┘
```

---

# 🔄 Workflow

```text
1. Dataset Preparation
        ↓
2. Data Validation
        ↓
3. Image Preprocessing
        ↓
4. Normal Image Selection
        ↓
5. CNN Feature Extraction
        ↓
6. PatchCore Memory Bank Construction
        ↓
7. Anomaly Score Calculation
        ↓
8. Threshold Calibration
        ↓
9. Image-Level Evaluation
        ↓
10. Pixel-Level Evaluation
        ↓
11. Failure Analysis
        ↓
12. ONNX Export
        ↓
13. ONNX Runtime Benchmark
        ↓
14. OpenVINO Benchmark
        ↓
15. TensorRT FP16 Benchmark
        ↓
16. INT8 Compression Analysis
        ↓
17. Feature Consistency Validation
        ↓
18. Production Monitoring
        ↓
19. Drift Detection
        ↓
20. Automated Testing
        ↓
21. Docker / Edge Deployment
```

---

# 🛠️ Tech Stack

## Programming

* Python 3.11
* NumPy
* Pandas

## Deep Learning

* PyTorch
* Torchvision

## Computer Vision

* OpenCV
* PIL

## Anomaly Detection

* PatchCore
* Nearest Neighbor Search

## Model Optimization

* ONNX
* ONNX Runtime
* OpenVINO
* TensorRT
* FP16
* INT8

## Monitoring

* PSI-based drift detection
* Prediction monitoring
* Anomaly-score monitoring
* Brightness monitoring
* Contrast monitoring
* Latency monitoring

## Testing

* PyTest

## Deployment

* Docker
* FastAPI
* Streamlit

## Dataset

* MVTec AD

---

# 📂 Project Structure

```text
edge_deployed/
│
├── data/
│   └── mvtec_anomaly_detection/
│       └── bottle/
│           ├── train/
│           │   └── good/
│           │
│           └── test/
│               ├── good/
│               ├── broken_large/
│               ├── broken_small/
│               └── contamination/
│
├── models/
│   ├── resnet18_feature_extractor.onnx
│   ├── resnet18_feature_extractor_fp16.engine
│   └── ...
│
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── patchcore/
│   ├── inference/
│   ├── evaluation/
│   ├── monitoring/
│   ├── deployment/
│   └── utils/
│
├── tests/
│   ├── test_feature_extractor.py
│   ├── test_patchcore.py
│   ├── test_preprocessing.py
│   └── ...
│
├── outputs/
│   ├── benchmark/
│   ├── evaluation/
│   ├── monitoring/
│   ├── predictions/
│   └── reports/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📋 Prerequisites

Recommended environment:

```text
Python >= 3.11
CUDA-compatible NVIDIA GPU
CUDA Toolkit
cuDNN
TensorRT
OpenVINO
ONNX Runtime
Docker Desktop
Git
```

For CPU-only execution, GPU and TensorRT are not required.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd edge_deployed
```

## 2. Create virtual environment

```bash
python -m venv .venv
```

## 3. Activate environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

## 6. Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 🔑 Environment Variables

The project can be configured using environment variables.

Example:

```env
MODEL_PATH=models/resnet18_feature_extractor.onnx
CATEGORY=bottle
DEVICE=cpu
ANOMALY_THRESHOLD=17.364517
LOG_LEVEL=INFO
```

For GPU deployment:

```env
DEVICE=cuda
```

For production systems, secrets and infrastructure credentials should be provided through environment variables rather than hard-coded in source code.

---

# ▶️ Usage

## Run PatchCore inference

```bash
python -m src.patchcore.inference
```

## Run evaluation

```bash
python -m src.evaluation
```

## Run failure analysis

```bash
python -m src.evaluation.failure_analysis
```

## Run ONNX Runtime benchmark

```bash
python -m src.benchmark.onnx_benchmark
```

## Run OpenVINO benchmark

```bash
python -m src.benchmark.openvino_benchmark
```

## Run TensorRT benchmark

```bash
python -m src.benchmark.tensorrt_benchmark
```

## Run monitoring

```bash
python -m src.monitoring.monitor
```

## Run Streamlit UI

If `app.py` is located in the project root:

```bash
streamlit run app.py
```

The application will be available locally through the Streamlit server.

---

# 💡 Example

Example inference result:

```json
{
    "filename": "000.png",
    "category": "bottle",
    "prediction": "DEFECT",
    "risk_level": "MEDIUM",
    "action": "HOLD_FOR_INSPECTION",
    "anomaly_score": 28.195,
    "threshold": 17.364517,
    "latency_ms": 258.11,
    "device": "cuda"
}
```

Example normal prediction:

```json
{
    "filename": "019.png",
    "category": "bottle",
    "prediction": "NORMAL",
    "anomaly_score": 13.338,
    "threshold": 17.364517
}
```

Decision logic:

```text
anomaly_score < threshold
        │
        └── NORMAL

anomaly_score >= threshold
        │
        └── DEFECT
```

Production threshold:

```text
17.364517
```

---

# 🧪 Testing

Automated tests were implemented for the main components.

Run the complete test suite:

```bash
python -m pytest tests -q
```

Current result:

```text
4 passed
```

Individual tests:

```bash
python -m pytest tests/test_feature_extractor.py -q
python -m pytest tests/test_patchcore.py -q
python -m pytest tests/test_preprocessing.py -q
```

Test coverage includes:

* Feature extractor behavior
* PatchCore functionality
* Image preprocessing
* Model/inference components

---

# 📊 Evaluation

## Image-Level PatchCore Evaluation

The final PatchCore evaluation produced:

| Metric              |     Result |
| ------------------- | ---------: |
| Image AUROC         | **0.9968** |
| Accuracy            | **0.9759** |
| Precision           | **1.0000** |
| Recall              | **0.9683** |
| F1 Score            | **0.9839** |
| False Positive Rate | **0.0000** |
| True Negatives      |     **20** |
| False Positives     |      **0** |
| False Negatives     |      **2** |
| True Positives      |     **61** |
| Evaluation Images   |     **83** |

Confusion matrix:

```text
                Predicted
              Normal  Defect
Actual Normal   20      0
Actual Defect    2     61
```

The evaluation used:

```text
Normal images : 20
Defect images : 63
Total images  : 83
```

---

# 🎯 Threshold Calibration

Production threshold:

```text
17.364517
```

Youden threshold:

```text
17.7677
```

The production threshold was selected for the deployed pipeline and used consistently across inference and monitoring.

---

# 🖼️ Pixel-Level Evaluation

Pixel-level localization results:

| Metric          |     Result |
| --------------- | ---------: |
| Pixel AUROC     | **0.9615** |
| Pixel Precision | **0.6584** |
| Pixel Recall    | **0.4301** |
| Pixel F1        | **0.5204** |
| Pixel IoU       | **0.3517** |

Pixel-level metrics evaluate how accurately the system localizes defective regions rather than only deciding whether the complete image is defective.

---

# 📈 PRO Evaluation

The Per-Region Overlap evaluation produced:

```text
PRO-AUC : 0.8899
FPR range tested : 0.00 - 0.30
```

PRO evaluation provides an additional measure of region-level anomaly localization quality.

---

# 🔍 Failure Analysis

Failure analysis was performed to identify false negatives, false positives, and difficult examples.

Final summary:

```text
Total images   : 83
Normal images  : 20
Defect images  : 63
False positives: 0
False negatives: 2
```

Defect-type analysis:

| Defect Type   | Count | FN |
| ------------- | ----: | -: |
| broken_large  |    20 |  0 |
| broken_small  |    22 |  0 |
| contamination |    21 |  2 |

The false negatives occurred in the **contamination** category, indicating that this defect type requires additional attention for future robustness improvements.

---

# ⚡ Performance

## CPU Baseline

| Metric          |              Result |
| --------------- | ------------------: |
| Average latency |       **172.45 ms** |
| P50 latency     |       **153.27 ms** |
| P99 latency     |       **315.48 ms** |
| Minimum latency |        **94.17 ms** |
| Maximum latency |       **332.71 ms** |
| FPS             |            **5.80** |
| Throughput      | **5.80 images/sec** |
| Memory change   |         **5.06 MB** |

---

## PyTorch CPU vs ONNX Runtime CPU

| Runtime          | Average Latency |        FPS |
| ---------------- | --------------: | ---------: |
| PyTorch CPU      |        19.67 ms |      50.84 |
| ONNX Runtime CPU |     **9.47 ms** | **105.64** |

Measured ONNX Runtime speedup:

```text
2.08x
```

Feature consistency:

```text
Layer2 max absolute difference : 0.00000626
Layer3 max absolute difference : 0.00000930
```

This demonstrates close numerical agreement between the PyTorch and ONNX Runtime feature extraction outputs.

---

# 🚀 OpenVINO Performance

OpenVINO CPU benchmark:

| Metric          |       Result |
| --------------- | -----------: |
| Average latency | **13.87 ms** |
| P50 latency     | **13.14 ms** |
| P95 latency     | **20.80 ms** |
| Minimum latency |  **8.61 ms** |
| Maximum latency | **26.17 ms** |
| FPS             |    **72.09** |

---

# ⚡ TensorRT FP16 Performance

TensorRT FP16 engine:

```text
Engine size : 8.01 MB
```

Benchmark:

| Metric     |      Result |
| ---------- | ----------: |
| Latency    | **0.91 ms** |
| FPS        | **1093.03** |
| Warmup     |          20 |
| Iterations |         100 |

The TensorRT FP16 engine was successfully loaded and benchmarked on an NVIDIA GPU.

---

# 📦 Model Compression

FP32 ONNX model:

```text
10.61 MB
```

INT8 ONNX model:

```text
2.69 MB
```

Size reduction:

```text
74.66%
```

Compression:

```text
3.95x
```

---

# 🔬 FP32 vs FP16 Feature Consistency

TensorRT FP16 feature consistency was validated against FP32 inference.

### Layer 2

```text
Max absolute error : 0.01247847
Mean absolute error: 0.00072596
Cosine similarity  : 0.99998969
```

### Layer 3

```text
Max absolute error : 0.00815517
Mean absolute error: 0.00049239
Cosine similarity  : 0.99998742
```

Final result:

```text
FP32 vs FP16 consistency: PASSED
```

---

# 🔬 FP32 vs INT8 Feature Consistency

### Output 0

```text
Max absolute error : 0.38782989
Mean absolute error: 0.02013860
Cosine similarity  : 0.99198960
```

### Output 1

```text
Max absolute error : 0.48990583
Mean absolute error: 0.01477477
Cosine similarity  : 0.99273813
```

Final assessment:

```text
Feature consistency: REVIEW REQUIRED
INT8 deployment assessment: REQUIRES REVIEW
```

Therefore, INT8 compression provides significant model-size reduction, but the current feature consistency results require further validation before using INT8 as the primary production configuration.

---

# 📊 Monitoring & Drift Detection

The monitoring system tracks:

* Anomaly score distribution
* Prediction distribution
* Brightness
* Contrast
* Latency
* FPS
* Population Stability Index (PSI)
* Overall deployment status

Example monitoring result:

```text
ANOMALY SCORE
Baseline mean : 16.0050
Current mean  : 16.4300
PSI           : 0.14907
Status        : MODERATE_DRIFT
```

```text
PREDICTIONS
Threshold     : 17.364517
Normal        : 12
Defect        : 8
Defect rate   : 0.4000
```

```text
BRIGHTNESS
Baseline mean : 127.3500
Current mean  : 128.3500
PSI           : 0.1648
Status        : MODERATE_DRIFT
```

```text
CONTRAST
Baseline mean : 44.5000
Current mean  : 45.0000
PSI           : 0.0549
Status        : NO_DRIFT
```

Latency monitoring:

```text
Mean : 90.63 ms
P50  : 90.85 ms
P95  : 93.88 ms
P99  : 94.14 ms
FPS  : 11.03
```

Overall monitoring status:

```text
WARNING
```

This demonstrates how an edge deployment can continuously monitor both model behavior and input-data characteristics.

---

# 📸 Screenshots / Demo

## PatchCore Evaluation

![PatchCore Metrics](docs/results/patchcore_metrics.png)

---

## Pixel-Level Evaluation

![Pixel Evaluation](docs/results/pixel_evaluation.png)

---

## PRO Evaluation

![PRO Evaluation](docs/results/pro_eval.png)

---

## Failure Analysis

![Failure Analysis](docs/results/failure_analysis.png)

---

## CPU Baseline Benchmark

![CPU Baseline](docs/results/cpu_baseline.png)

---

## ONNX Runtime Benchmark

![ONNX Runtime](docs/results/ONNX_runtime.png)

---

## OpenVINO CPU Benchmark

![OpenVINO Results](docs/results/OPENVIO_result.png)

---

## TensorRT FP32 vs FP16

![FP32 vs FP16](docs/results/fp32_vs_fp16.png)

---

## FP32 vs INT8

![FP32 vs INT8](docs/results/fp32_vs_int8.png)

---

## Monitoring & Drift Detection

![Monitoring and Drift](docs/results/monitoring_drift.png)

---

## Automated Tests

![PyTest Results](docs/results/py_test.png)

---

# 🐳 Docker Deployment

The application can be containerized for reproducible deployment.

Build the Docker image:

```bash
docker build -t edge-defect-detection .
```

Run:

```bash
docker run -p 8501:8501 edge-defect-detection
```

For GPU-enabled deployment, the Docker configuration can be extended with NVIDIA Container Toolkit support.

---

# 🔐 Security

Security considerations for production deployment include:

* Do not hard-code credentials.
* Use environment variables for secrets.
* Restrict API access.
* Validate uploaded images.
* Limit image size and request payloads.
* Validate model paths.
* Restrict filesystem access.
* Run containers with minimal privileges.
* Avoid exposing internal model paths.
* Implement request logging and monitoring.
* Add authentication and authorization for production APIs.
* Use HTTPS when deployed outside a trusted network.

---

# 🔮 Future Improvements

## Model Improvements

* Improve contamination defect recall.
* Add additional industrial categories.
* Evaluate FastFlow.
* Evaluate PaDiM and other anomaly detection methods.
* Ensemble multiple anomaly detection models.
* Improve pixel-level localization.
* Optimize anomaly threshold per product category.

## Robustness

* Lighting variation augmentation.
* Camera-position variation.
* Product appearance variation.
* Domain adaptation.
* Test-time adaptation.
* Environmental robustness evaluation.

## Edge Optimization

* Further TensorRT optimization.
* INT8 calibration.
* INT8 accuracy validation.
* CUDA optimization.
* TensorRT dynamic batching.
* GPU memory optimization.
* Multi-camera inference.
* Distributed edge inference.

## Production

* Real-time camera streaming.
* FastAPI inference service.
* Streamlit monitoring dashboard.
* Centralized logging.
* Prometheus/Grafana monitoring.
* Model versioning.
* Automated model update pipeline.
* Continuous drift monitoring.
* Alerting system.
* Edge-device fleet management.

## Advanced AI

* Few-shot defect adaptation.
* Continual learning.
* Automated threshold recalibration.
* Explainable anomaly heatmaps.
* Defect severity estimation.
* Defect classification after anomaly detection.

---

# ⚠️ Limitations

* Current evaluation focuses primarily on the MVTec AD `bottle` category.
* The dataset does not represent every real manufacturing environment.
* The current contamination category produced the observed false negatives.
* Pixel-level recall is lower than image-level classification recall.
* INT8 feature consistency currently requires additional validation.
* Real production camera streams may introduce lighting and viewpoint changes not fully represented in the benchmark.
* CPU and GPU performance will vary depending on hardware.
* TensorRT performance is hardware-dependent.
* Production deployment requires additional security, observability, and infrastructure configuration.
* Thresholds may need recalibration when the product, camera, or environment changes.

---

# 🤝 Contributing

Contributions are welcome.

Recommended workflow:

```text
1. Fork the repository
2. Create a feature branch
3. Implement the change
4. Add/update tests
5. Run the test suite
6. Commit the changes
7. Open a Pull Request
```

Example:

```bash
git checkout -b feature/new-anomaly-model
```

Run tests:

```bash
python -m pytest tests -q
```

---

# 📄 License

This project is available under the MIT License.

Add the complete license text to:

```text
LICENSE
```

if you choose to publish the repository under MIT.

---

# 👨‍💻 Author

**Naga Ganesh**

AI / Machine Learning / Computer Vision / Edge AI

This project demonstrates an end-to-end approach to building, evaluating, optimizing, monitoring, and deploying an industrial anomaly detection system for edge environments.

---

# ⭐ Key Results

```text
============================================================
                FINAL PROJECT RESULTS
============================================================

PATCHCORE
------------------------------------------------------------
Image AUROC          : 0.9968
Accuracy             : 0.9759
Precision            : 1.0000
Recall               : 0.9683
F1 Score             : 0.9839
FPR                  : 0.0000

PIXEL-LEVEL
------------------------------------------------------------
Pixel AUROC          : 0.9615
Pixel Precision      : 0.6584
Pixel Recall         : 0.4301
Pixel F1             : 0.5204
Pixel IoU            : 0.3517

PRO
------------------------------------------------------------
PRO-AUC              : 0.8899

ONNX RUNTIME
------------------------------------------------------------
CPU Latency          : 9.47 ms
CPU FPS              : 105.64
Speedup vs PyTorch   : 2.08x

OPENVINO
------------------------------------------------------------
CPU Latency          : 13.87 ms
CPU FPS              : 72.09

TENSORRT FP16
------------------------------------------------------------
Latency              : 0.91 ms
FPS                  : 1093.03
Engine Size          : 8.01 MB

MODEL COMPRESSION
------------------------------------------------------------
FP32 ONNX            : 10.61 MB
INT8 ONNX            : 2.69 MB
Size Reduction       : 74.66%
Compression          : 3.95x

MONITORING
------------------------------------------------------------
Anomaly PSI          : 0.14907
Brightness PSI       : 0.1648
Contrast PSI         : 0.0549
Overall Status       : WARNING

TESTING
------------------------------------------------------------
Automated Tests      : 4 PASSED
============================================================
```

---

# 📌 Project Highlights

```text
✓ Unseen industrial defect detection
✓ PatchCore anomaly detection
✓ Image-level evaluation
✓ Pixel-level localization
✓ Threshold calibration
✓ Failure analysis
✓ ONNX deployment
✓ OpenVINO optimization
✓ TensorRT FP16 acceleration
✓ INT8 compression analysis
✓ Feature consistency validation
✓ CPU/GPU benchmarking
✓ Production monitoring
✓ Data drift detection
✓ Automated testing
✓ Docker-ready deployment
✓ Edge AI architecture
```

---

# 🏭 Production-Oriented Pipeline

```text
                 CAMERA / IMAGE SOURCE
                          │
                          ▼
                ┌──────────────────┐
                │ PREPROCESSING    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ FEATURE EXTRACTOR│
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    PATCHCORE     │
                │  MEMORY BANK     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ ANOMALY SCORE    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ THRESHOLD        │
                │ 17.364517        │
                └────────┬─────────┘
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
               NORMAL         DEFECT
                  │             │
                  └──────┬──────┘
                         │
                         ▼
                ┌──────────────────┐
                │ MONITORING       │
                │ DRIFT DETECTION  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ EDGE DEPLOYMENT  │
                │ ONNX / OpenVINO  │
                │ TensorRT         │
                └──────────────────┘
```

---

# 🚀 Conclusion

This project demonstrates a complete **industrial computer vision and edge AI pipeline**, starting from anomaly detection and evaluation and extending through model optimization, runtime benchmarking, monitoring, drift detection, testing, and deployment.

The final system achieved a **0.9968 image-level AUROC** and **0.9839 F1 score** on the evaluated MVTec AD test set, while the deployment experiments demonstrated substantial inference acceleration through ONNX Runtime, OpenVINO, and TensorRT FP16.

The project also explicitly evaluates the trade-off between **model compression, numerical consistency, inference latency, throughput, and production reliability**, making it suitable as a practical portfolio project for **Computer Vision, Deep Learning, Edge AI, and ML Engineering roles**.

````

### One important GitHub cleanup

For the screenshots, I recommend this exact repository layout:

```text
edge_deployed/
│
├── docs/
│   └── results/
│       ├── cpu_baseline.png
│       ├── failure_analysis.png
│       ├── fp32_vs_fp16.png
│       ├── fp32_vs_int8.png
│       ├── monitoring_drift.png
│       ├── ONNX_runtime.png
│       ├── OPENVIO_result.png
│       ├── patchcore_metrics.png
│       ├── pixel_evaluation.png
│       ├── pro_eval.png
│       └── py_test.png
│
├── src/
├── tests/
├── models/
├── outputs/
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
````

This makes the README's screenshot paths work cleanly on GitHub:

```text
docs/results/patchcore_metrics.png
docs/results/ONNX_runtime.png
docs/results/OPENVIO_result.png
...
```

**Your strongest numbers to highlight near the top of the GitHub README are:** **0.9968 AUROC**, **0.9839 F1**, **0% FPR**, **105.64 FPS ONNX CPU**, **72.09 FPS OpenVINO CPU**, and **1093 FPS TensorRT FP16**.
