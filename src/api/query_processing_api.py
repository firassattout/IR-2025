from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from src.processing_ranking.query_processing import retrieve_documents_tfidf, retrieve_documents_bert, retrieve_documents_hybrid
from src.representation.tfidf_model import load_tfidf_model
from src.representation.bert_model import load_bert_embeddings
from src.representation.vector_store import load_faiss_index, retrieve_documents_faiss
import joblib
import os

app = FastAPI(title="Query Processing API")

class RetrieveTFIDFRequest(BaseModel):
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: Optional[str] = None
    query: str
    index_dir: Optional[str] = None
    k: Optional[int] = 10
    candidate_limit: Optional[int] = 100

class RetrieveBERTRequest(BaseModel):
    bert_model_path: str
    bert_vector_path: str
    query: str
    index_dir: Optional[str] = None
    k: Optional[int] = 10
    candidate_limit: Optional[int] = 100

class RetrieveHybridRequest(BaseModel):
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: Optional[str] = None
    bert_model_path: str
    bert_vector_path: str
    hybrid_vector_path: str
    query: str
    index_dir: Optional[str] = None
    alpha: Optional[float] = 0.5
    k: Optional[int] = 10
    candidate_limit: Optional[int] = 100

class RetrieveFAISSRequest(BaseModel):
    dataset_name: str
    index_path: str
    vector_path: str
    model_type: str  # 'faiss_tfidf', 'faiss_bert', 'faiss_hybrid'
    query: str
    svd_path: Optional[str] = None
    bert_model_path: Optional[str] = None
    index_dir: Optional[str] = None
    k: Optional[int] = 10
    candidate_limit: Optional[int] = 100
    alpha: Optional[float] = 0.5

@app.post("/retrieve_documents_tfidf")
async def api_retrieve_documents_tfidf(data: RetrieveTFIDFRequest):
    try:
        result = load_tfidf_model(data.tfidf_model_path, data.tfidf_vector_path, data.svd_path)
        if data.svd_path:
            tfidf_vectorizer, tfidf_matrix, svd_model, tfidf_doc_ids = result
        else:
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = result
            svd_model = None
        top_docs, cleaned_query = retrieve_documents_tfidf(
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids, data.query, svd_model, data.index_dir, data.k, data.candidate_limit
        )
        top_docs_response = [{"doc_id": doc_id, "similarity": float(similarity)} for doc_id, similarity in top_docs]
        return {
            "cleaned_query": cleaned_query,
            "top_documents": top_docs_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TF-IDF retrieval failed: {str(e)}")

@app.post("/retrieve_documents_bert")
async def api_retrieve_documents_bert(data: RetrieveBERTRequest):
    try:
        tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(data.bert_model_path, data.bert_vector_path)
        top_docs, cleaned_query = retrieve_documents_bert(
            tokenizer, bert_model, bert_embeddings, bert_doc_ids, data.query, data.index_dir, data.k, data.candidate_limit
        )
        top_docs_response = [{"doc_id": doc_id, "similarity": float(similarity)} for doc_id, similarity in top_docs]
        return {
            "cleaned_query": cleaned_query,
            "top_documents": top_docs_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BERT retrieval failed: {str(e)}")

@app.post("/retrieve_documents_hybrid")
async def api_retrieve_documents_hybrid(data: RetrieveHybridRequest):
    try:
        result_tfidf = load_tfidf_model(data.tfidf_model_path, data.tfidf_vector_path, data.svd_path)
        if data.svd_path:
            tfidf_vectorizer, tfidf_matrix, svd_model, tfidf_doc_ids = result_tfidf
        else:
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = result_tfidf
            svd_model = None
        bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(data.bert_model_path, data.bert_vector_path)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hybrid_vector_path = os.path.join(project_root, data.hybrid_vector_path)
        hybrid_vectors = joblib.load(hybrid_vector_path)
        hybrid_doc_ids = joblib.load(f"{hybrid_vector_path}_doc_ids")
        top_docs, cleaned_query = retrieve_documents_hybrid(
            tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, hybrid_vectors, hybrid_doc_ids, data.query, data.index_dir, data.alpha, data.k, data.candidate_limit
        )
        top_docs_response = [{"doc_id": doc_id, "similarity": float(similarity)} for doc_id, similarity in top_docs]
        return {
            "cleaned_query": cleaned_query,
            "top_documents": top_docs_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid retrieval failed: {str(e)}")

@app.post("/retrieve_documents_faiss")
async def api_retrieve_documents_faiss(data: RetrieveFAISSRequest):
    try:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
        index_path = os.path.join(project_root, data.index_path)
        vector_path = os.path.join(project_root, data.vector_path)
        svd_path = os.path.join(project_root, data.svd_path) if data.svd_path else None
        bert_model_path = os.path.join(project_root, data.bert_model_path) if data.bert_model_path else None
        index_dir = os.path.join(project_root, data.index_dir) if data.index_dir else None

        faiss_index, doc_ids = load_faiss_index(index_path)

        if data.model_type == 'faiss_tfidf':
            if not svd_path:
                raise HTTPException(status_code=400, detail="svd_path is required for FAISS TF-IDF")
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(
                f"models/tfidf/{data.dataset_name}_tfidf.joblib",
                vector_path
            )
            svd_model = joblib.load(svd_path)
            vectorizer_or_tokenizer = tfidf_vectorizer
            model = None
            extra_params = {"svd_model": svd_model}
        elif data.model_type == 'faiss_bert':
            if not bert_model_path:
                raise HTTPException(status_code=400, detail="bert_model_path is required for FAISS BERT")
            tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
                bert_model_path, vector_path
            )
            vectorizer_or_tokenizer = tokenizer
            model = bert_model
            extra_params = {}
        elif data.model_type == 'faiss_hybrid':
            if not (svd_path and bert_model_path):
                raise HTTPException(status_code=400, detail="Both svd_path and bert_model_path are required for FAISS Hybrid")
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(
                f"models/tfidf/{data.dataset_name}_tfidf.joblib",
                vector_path
            )
            svd_model = joblib.load(svd_path)
            tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
                bert_model_path, vector_path
            )
            vectorizer_or_tokenizer = tfidf_vectorizer
            model = None
            extra_params = {"svd_model": svd_model, "bert_tokenizer": tokenizer, "bert_model": bert_model, "alpha": data.alpha}
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Choose 'faiss_tfidf', 'faiss_bert', or 'faiss_hybrid'.")

        top_docs, cleaned_query = retrieve_documents_faiss(
            vectorizer_or_tokenizer=vectorizer_or_tokenizer,
            model=model,
            vectors_or_index=faiss_index,
            doc_ids=doc_ids,
            query=data.query,
            index_dir=index_dir,
            model_type=data.model_type.split('_')[1],
            k=data.k,
            candidate_limit=data.candidate_limit,
            **extra_params
        )

        top_docs_response = [{"doc_id": doc_id, "similarity": float(similarity)} for doc_id, similarity in top_docs]
        return {
            "cleaned_query": cleaned_query,
            "top_documents": top_docs_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS retrieval failed: {str(e)}")