FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install core PyTorch stack (use pre-built, no extra index)
RUN pip install --no-cache-dir \
    torchvision==0.18.0 \
    torchaudio==2.3.0

# Training stack — use >= for flexibility, pip resolves compatible versions
RUN pip install --no-cache-dir \
    "transformers>=4.45.0" \
    "peft>=0.13.0" \
    "trl>=0.12.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.44.0" \
    "datasets>=2.21.0" \
    "huggingface_hub>=0.26.0" \
    "tensorboard>=2.18.0"

# Google Cloud — use >= for flexible version resolution
# Exact pins cause build failures when versions don't exist in all envs
RUN pip install --no-cache-dir \
    google-cloud-storage>=2.16.0 \
    google-cloud-aiplatform>=1.60.0 \
    vertexai>=1.60.0

# Utilities
RUN pip install --no-cache-dir \
    tqdm \
    pyyaml \
    safetensors

COPY train.py /app/train.py
COPY requirements.txt /app/requirements.txt
COPY config /app/config

ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python", "/app/train.py"]
