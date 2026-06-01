# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY main.py ./
COPY server.py ./

RUN uv pip install --system --no-cache .

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('google/embeddinggemma-300m'); \
_ = m.encode_document(['warmup']); \
print('weights ready, dim:', m.get_sentence_embedding_dimension())"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
