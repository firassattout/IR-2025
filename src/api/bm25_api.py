from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.representation.bm25_model import build_bm25_model, load_bm25_model, transform_bm25_query
import os
import numpy as np

app = FastAPI(title="BM25 API")

class BuildModelRequest(BaseModel):
    collection_name: str
    model_path: str
    vector_path: str

class LoadModelRequest(BaseModel):
    model_path: str
    vector_path: str

class TransformQueryRequest(BaseModel):
    model_path: str
    query_text: str
    vector_path: str

@app.post("/build_bm25_model")
async def api_build_model(data: BuildModelRequest):
    build_bm25_model(data.collection_name, data.model_path, data.vector_path)
    return {"status": "Model built successfully"}


@app.post("/load_bm25_model")
async def api_load_bm25_model(data: LoadModelRequest):
    """
    تحميل نموذج BM25 ومعرفات الوثائق.
    """
    try:
        bm25, doc_ids = load_bm25_model(data.model_path, data.vector_path)
        return {
            "doc_ids": doc_ids,  # إرجاع معرفات الوثائق
            "status": "Model loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/transform_query")
async def api_transform_query(data: TransformQueryRequest):
    """
    تحويل الاستعلام إلى درجات BM25.
    """
    try:
        # تحميل النموذج
        bm25, doc_ids = load_bm25_model(data.model_path, data.vector_path)

        # تحويل الاستعلام
        scores, cleaned_query = transform_bm25_query(bm25, data.query_text)
        return {
            "cleaned_text": cleaned_query,
            "scores_shape": str(scores.shape),  # شكل مصفوفة الدرجات
            "scores": scores.tolist()  # تحويل إلى قائمة
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transforming query: {str(e)}")