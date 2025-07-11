from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.representation.vector_store import build_faiss_index
from src.representation.tfidf_model import load_tfidf_model
from src.representation.bert_model import load_bert_embeddings
import os
import joblib
import numpy as np

app = FastAPI(title="FAISS Build API")

class BuildFAISSRequest(BaseModel):
    dataset: str
    model_type: str  # 'tfidf', 'bert', or 'hybrid'
    vector_path: str
    index_path: str
    svd_path: Optional[str] = None  # لـ TF-IDF وHybrid
    bert_model_path: Optional[str] = None  # لـ BERT وHybrid
    tfidf_model_path: Optional[str] = None  # لـ BERT وHybrid

@app.post("/build_faiss_index")
async def api_build_faiss_index(data: BuildFAISSRequest):
    """
    بناء فهرس FAISS لمجموعة بيانات ونموذج محدد.
    """
    try:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
        vector_path = os.path.join(project_root, data.vector_path)
        index_path = os.path.join(project_root, data.index_path)
        svd_path = os.path.join(project_root, data.svd_path) if data.svd_path else None
        bert_model_path = os.path.join(project_root, data.bert_model_path) if data.bert_model_path else None
        tfidf_model_path = os.path.join(project_root, data.tfidf_model_path) if data.tfidf_model_path else None

        # تحميل المتجهات بناءً على نوع النموذج
        if data.model_type == 'tfidf':
            if not svd_path:
                raise HTTPException(status_code=400, detail="svd_path is required for TF-IDF model")
            vectorizer, tfidf_matrix, doc_ids = load_tfidf_model(
                tfidf_model_path,
                vector_path
            )
            svd_model = joblib.load(svd_path)
            vectors = svd_model.transform(tfidf_matrix).astype(np.float32)
        elif data.model_type == 'bert':
            if not bert_model_path:
                raise HTTPException(status_code=400, detail="bert_model_path is required for BERT model")
            tokenizer, bert_model, vectors, doc_ids = load_bert_embeddings(
                bert_model_path,
                vector_path
            )
            vectors = vectors.astype(np.float32)
        elif data.model_type == 'hybrid':
            if not (svd_path and bert_model_path):
                raise HTTPException(status_code=400, detail="Both svd_path and bert_model_path are required for Hybrid model")
            vectors = joblib.load(vector_path)
            doc_ids = joblib.load(f"{vector_path}_doc_ids")
            vectors = vectors.astype(np.float32)
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'tfidf', 'bert', or 'hybrid'.")

        # بناء فهرس FAISS
        build_faiss_index(vectors, doc_ids, index_path)
        return {"status": f"FAISS index built successfully for {data.dataset} ({data.model_type})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))