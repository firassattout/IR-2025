from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.representation.tfidf_model import build_tfidf_model, transform_tfidf_query, load_tfidf_model
import os
import numpy as np

app = FastAPI(title="tfidf API")

class BuildModelRequest(BaseModel):
    collection_name: str
    model_path: str
    vector_path: str
    svd_path: Optional[str] = None
    n_components: Optional[int] = 384

class LoadModelRequest(BaseModel):
    model_path: str
    vector_path: str
    svd_path: Optional[str] = None

class TransformQueryRequest(BaseModel):
    model_path: str
    query_text: str
    vector_path: str
    svd_path: Optional[str] = None

@app.post("/build_tfidf_model")
async def api_build_model(data: BuildModelRequest):
        build_tfidf_model(data.collection_name, data.model_path, data.vector_path, data.svd_path, data.n_components)
        return {"status": "Model built successfully"}


@app.post("/load_tfidf_model")
async def api_load_tfidf_model(data: LoadModelRequest):
        # تحميل النموذج
        result = load_tfidf_model(data.model_path, data.vector_path, data.svd_path)
        if data.svd_path:
            vectorizer, tfidf_matrix, svd, doc_ids = result
        else:
            vectorizer, tfidf_matrix, doc_ids = result

        feature_names = vectorizer.get_feature_names_out()  # تحويل إلى قائمة
        return {
            "vectorizerQ": feature_names.tolist(),

        }


@app.post("/transform_query")
async def api_transform_query(data: TransformQueryRequest):
        # تحميل النموذج
        result = load_tfidf_model(data.model_path, data.vector_path, data.svd_path)
        if data.svd_path:
            vectorizer, tfidf_matrix, svd, doc_ids = result
        else:
            vectorizer, tfidf_matrix, doc_ids = result

        # تحويل الاستعلام
        vec, cleaned = transform_tfidf_query(vectorizer, data.query_text)
        queryVec=vec.toarray()[0]
        return {
            "cleaned_text": cleaned,
            "vector_shape": str(vec.shape),  # تحويل tuple إلى سلسلة
            "query_vec": queryVec.tolist()  # تحويل إلى قائمة
        }
