# CUDA Base-Image mit Ubuntu 22.04
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Python 3.12 installieren
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    && ln -s /usr/bin/python3.12 /usr/bin/python3 \
    && apt-get install -y python3-pip \
    && apt-get clean

# Arbeitsverzeichnis im Container
WORKDIR /workspace

# requirements.txt & Code kopieren
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Optional: Standardbefehl
CMD ["python3", "main.py"]
