
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.representation.bert_model import build_bert_embeddings, load_bert_embeddings, transform_bert_query
import os
import numpy as np

app = FastAPI(title="BERT API")

class BuildBertRequest(BaseModel):
    collection_name: str
    model_path: str
    vector_path: str
    batch_size: Optional[int] = 16

class LoadBertRequest(BaseModel):
    model_path: str
    vector_path: str

class TransformBertQueryRequest(BaseModel):
    model_path: str
    vector_path: str
    query_text: str

@app.post("/build_bert_embeddings")
async def api_build_bert_embeddings(data: BuildBertRequest):
        build_bert_embeddings(data.collection_name, data.model_path, data.vector_path, data.batch_size)
        return {"status": "BERT embeddings built successfully"}


@app.post("/load_bert_embeddings")
async def api_load_bert_embeddings(data: LoadBertRequest):
        tokenizer, model, embeddings, doc_ids = load_bert_embeddings(data.model_path, data.vector_path)
        return {
            "status": "BERT embeddings loaded successfully",
            "embeddings_shape": str(embeddings.shape),
            "doc_ids_count": len(doc_ids)
        }

@app.post("/transform_bert_query")
async def api_transform_bert_query(data: TransformBertQueryRequest):
        tokenizer, model, _, _ = load_bert_embeddings(data.model_path, data.vector_path)
        query_embedding,_ = transform_bert_query(tokenizer, model, data.query_text)
        return {
            "query_embedding": query_embedding.tolist() 
        }
