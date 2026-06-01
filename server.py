from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MAX_CHUNK_WORDS = 1000
CHUNK_OVERLAP_WORDS = 200

model: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("google/embeddinggemma-300m", local_files_only=True)
    yield
    model = None


app = FastAPI(title="Embedding Inference Server", lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    input: str | list[str] = Field(..., description="Text or list of texts to embed")
    task_type: Literal["query", "document"] = Field(
        "document", description="Encoding strategy to use"
    )


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    chunk_count: int | None = Field(
        None, description="Number of chunks used (only present for long texts)"
    )
    original_length: int | None = Field(
        None, description="Original word count (only present for long texts)"
    )


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]


def chunk_text(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def embed_chunks(chunks: list[str], task_type: str) -> torch.Tensor:
    assert model is not None
    encode_fn = (
        model.encode_query if task_type == "query" else model.encode_document
    )
    result = encode_fn(chunks)
    return torch.as_tensor(result)


def aggregate_embeddings(chunk_embeddings: torch.Tensor) -> list[float]:
    t = torch.as_tensor(chunk_embeddings)
    mean_embedding = t.mean(dim=0)
    norm = torch.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    return mean_embedding.tolist()


def embed_text(text: str, task_type: str = "document") -> tuple[list[float], int | None, int | None]:
    words = text.split()
    original_length = len(words)

    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks, task_type)
    embedding = aggregate_embeddings(chunk_embeddings)

    chunk_count = len(chunks) if len(chunks) > 1 else None
    orig_len = original_length if len(chunks) > 1 else None

    return embedding, chunk_count, orig_len


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = (
        [request.input] if isinstance(request.input, str) else request.input
    )

    results = []
    for i, text in enumerate(inputs):
        embedding, chunk_count, original_length = embed_text(text, request.task_type)
        results.append(
            EmbeddingData(
                index=i,
                embedding=embedding,
                chunk_count=chunk_count,
                original_length=original_length,
            )
        )

    return EmbeddingResponse(model="google/embeddinggemma-300m", data=results)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
