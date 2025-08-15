FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# System deps needed by pandas/scipy etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Bring in the upstream code
RUN git clone https://github.com/IBM/diveye.git /workspace

# Serverless deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# Serverless handler
COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
