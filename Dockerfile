FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# The CPU PyTorch wheels link against libgomp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the CPU wheel first so the requirements install finds torch satisfied.
# Keep this version in sync with the torch pin in requirements.txt.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /app "$HF_HOME"
USER app

EXPOSE 8000

# The first start downloads the embedding model; the start period covers it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
