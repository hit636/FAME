FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential git curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/FAME
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
COPY . .
CMD ["bash", "scripts/00_smoke_test.sh"]
