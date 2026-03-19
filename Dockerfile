FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY train.py .
COPY train_wrapper.sh .
RUN chmod +x train_wrapper.sh
RUN mkdir -p /output

CMD ["./train_wrapper.sh"]
