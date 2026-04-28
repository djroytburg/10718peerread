FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.docker.txt

RUN python -m spacy download en_core_web_sm

COPY parse_pdfs_docling.py .
COPY parse_pdfs_parallel.py .
COPY anonymize_pdfs.py .
COPY anonymize_batch.py .
COPY convert_to_peerread_format.py .
COPY convert_json_to_markdown.py .
COPY create_balanced_split.py .
COPY diagnostics.py .
COPY validate_anonymization.py .
COPY scrape_neurips.py .
COPY PeerRead/ PeerRead/

VOLUME ["/workspace/input", "/workspace/output"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "parse_pdfs_docling.py", "--help"]
