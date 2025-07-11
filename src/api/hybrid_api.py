from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.representation.tfidf_model import load_tfidf_model, transform_tfidf_query
from src.representation.bert_model import load_bert_embeddings, transform_bert_query
from src.representation.hybrid_model import build_hybrid_model, transform_hybrid_query
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
app = FastAPI(title="Hybrid API")

class BuildHybridRequest(BaseModel):
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: str
    bert_model_path: str
    bert_vector_path: str
    hybrid_vector_path: str
    alpha: Optional[float] = 0.5
    n_components: Optional[int] = 384

class TransformHybridQueryRequest(BaseModel):
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: str
    bert_model_path: str
    bert_vector_path: str
    hybrid_vector_path: str
    query_text: str
    alpha: Optional[float] = 0.5

@app.post("/build_hybrid_model")
async def api_build_hybrid_model(data: BuildHybridRequest):
        build_hybrid_model(
            tfidf_model_path=data.tfidf_model_path,
            tfidf_vector_path=data.tfidf_vector_path,
            svd_path=data.svd_path,
            bert_model_path=data.bert_model_path,
            bert_vector_path=data.bert_vector_path,
            hybrid_vector_path=data.hybrid_vector_path,
            alpha=data.alpha,
            n_components=data.n_components
        )
        return {"status": "Hybrid model built successfully"}

@app.post("/transform_hybrid_query")
async def api_transform_hybrid_query(data: TransformHybridQueryRequest):

        result_tfidf = load_tfidf_model(data.tfidf_model_path, data.tfidf_vector_path, data.svd_path)
        if data.svd_path:
            tfidf_vectorizer, tfidf_matrix, svd_model, tfidf_doc_ids = result_tfidf
        else:
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = result_tfidf


        bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(data.bert_model_path, data.bert_vector_path)


        hybrid_query_vec,cleaned_text = transform_hybrid_query(
            tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, data.query_text, data.alpha
        )

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hybrid_vector_path = os.path.join(project_root, data.hybrid_vector_path)
        hybrid_vectors = joblib.load(hybrid_vector_path)
        hybrid_doc_ids = joblib.load(f"{hybrid_vector_path}_doc_ids")
        
        similarities = cosine_similarity(hybrid_query_vec, hybrid_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        top_docs = [{"doc_id": hybrid_doc_ids[idx], "similarity": float(similarities[idx])} for idx in top_indices]

        return {
            "query_vector_shape": str(hybrid_query_vec.shape),
            "query_vector": hybrid_query_vec.tolist(),
            "top_documents": top_docs
        }