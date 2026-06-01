from contextlib import asynccontextmanager
from typing import Literal, cast

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

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


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest):
    if model is None:
        return {"error": "Model not loaded"}

    inputs = (
        [request.input] if isinstance(request.input, str) else request.input
    )

    encode_fn = (
        model.encode_query if request.task_type == "query" else model.encode_document
    )
    embeddings = cast(torch.Tensor, encode_fn(inputs))

    return EmbeddingResponse(
        model="google/embeddinggemma-300m",
        data=[
            EmbeddingData(index=i, embedding=embeddings[i].tolist())
            for i in range(len(inputs))
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
