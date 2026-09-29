FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# APPLICATION
COPY src ./src
COPY config.yaml .
COPY configs ./configs

# DATA
COPY data ./data
# FASTAPI
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]