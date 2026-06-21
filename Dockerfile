FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller wheel, ~800 MB vs ~3 GB)
RUN pip install --no-cache-dir --root-user-action=ignore \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# torch/torchvision already installed above; skip them here
RUN pip install --no-cache-dir --root-user-action=ignore \
    $(grep -v -E '^torch' requirements.txt | tr '\n' ' ')

COPY app.py clip_score.py db.py ./
COPY static/ static/
COPY templates/ templates/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
