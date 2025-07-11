from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
import os
import sys
import joblib
import time
from src.representation.tfidf_model import load_tfidf_model
from src.representation.bert_model import load_bert_embeddings
from src.processing_ranking.query_processing import retrieve_documents_tfidf, retrieve_documents_bert, retrieve_documents_hybrid
from src.representation.vector_store import retrieve_documents_faiss, load_faiss_index

app = FastAPI(title="Frontend API")

# إعداد المسارات
project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
if project_root not in sys.path:
    sys.path.append(project_root)

# إعداد القوالب
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "../ui/templates"))

# إعداد MongoDB
client = MongoClient('localhost', 27017)
db = client['IR_PROJECT']

class SearchRequest(BaseModel):
    query: str
    model: str
    dataset: str
    k: Optional[int] = 10
    candidate_limit: Optional[int] = 100

loaded_models = {}

def load_models_once(dataset: str):
    if dataset in loaded_models:
        return loaded_models[dataset]
    
    models =load_models(dataset)
    loaded_models[dataset] = models
    return models

# دالة لتحميل النماذج بناءً على قاعدة البيانات
def load_models(dataset: str):
    tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(
        f"../models/tfidf/{dataset}_tfidf.joblib",
        f"../models/tfidf/{dataset}_tfidf_vectors.joblib"
    )
    svd_model = joblib.load(os.path.join(project_root, f"models/tfidf/{dataset}_svd.joblib"))
    bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
        f"../models/embeddings/{dataset}_bert.joblib",
        f"../models/embeddings/{dataset}_vectors.joblib"
    )
    hybrid_vectors = joblib.load(os.path.join(project_root, f"models/hybrid/{dataset}_hybrid_vectors.joblib"))
    hybrid_doc_ids = joblib.load(os.path.join(project_root, f"models/hybrid/{dataset}_hybrid_vectors.joblib_doc_ids"))
    tfidf_faiss_index, tfidf_faiss_doc_ids = load_faiss_index(os.path.join(project_root, f"models/faiss/{dataset}_tfidf_faiss.index"))
    bert_faiss_index, bert_faiss_doc_ids = load_faiss_index(os.path.join(project_root, f"models/faiss/{dataset}_bert_faiss.index"))
    hybrid_faiss_index, hybrid_faiss_doc_ids = load_faiss_index(os.path.join(project_root, f"models/faiss/{dataset}_hybrid_faiss.index"))
    index_dir = os.path.join(project_root, f"data/index_{dataset}")
    collection = db[dataset]
    return (
        tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids, svd_model,
        bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids,
        hybrid_vectors, hybrid_doc_ids,
        tfidf_faiss_index, tfidf_faiss_doc_ids,
        bert_faiss_index, bert_faiss_doc_ids,
        hybrid_faiss_index, hybrid_faiss_doc_ids,
        index_dir, collection
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """عرض صفحة الواجهة الرئيسية."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search(request: Request, query: str = Form(...), model: str = Form(...), dataset: str = Form(...), k: int = Form(10), candidate_limit: int = Form(100)):
    start_time = time.time()  # بدء المؤقت

    # تحميل النماذج وقاعدة البيانات
    (
        tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids, svd_model,
        bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids,
        hybrid_vectors, hybrid_doc_ids,
        tfidf_faiss_index, tfidf_faiss_doc_ids,
        bert_faiss_index, bert_faiss_doc_ids,
        hybrid_faiss_index, hybrid_faiss_doc_ids,
        index_dir, collection
    ) = load_models_once(dataset)

    # تنفيذ البحث بناءً على النموذج المختار
    if model == 'tfidf':
        results, cleaned_query = retrieve_documents_tfidf(
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids, query, svd_model=svd_model,
            index_dir=index_dir, k=k, candidate_limit=candidate_limit
        )
    elif model == 'bert':
        results, cleaned_query = retrieve_documents_bert(
            bert_tokenizer, bert_model, bert_embeddings, bert_doc_ids, query,
            index_dir=index_dir, k=k, candidate_limit=candidate_limit
        )
    elif model == 'hybrid':
        results, cleaned_query = retrieve_documents_hybrid(
            tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, hybrid_vectors, hybrid_doc_ids, query,
            index_dir=index_dir, alpha=0.5, k=k, candidate_limit=candidate_limit
        )
    elif model == 'faiss_tfidf':
        results, cleaned_query = retrieve_documents_faiss(
            tfidf_vectorizer, None, tfidf_faiss_index, tfidf_faiss_doc_ids, query,
            index_dir=index_dir, model_type='tfidf', svd_model=svd_model, k=k, candidate_limit=candidate_limit
        )
    elif model == 'faiss_bert':
        results, cleaned_query = retrieve_documents_faiss(
            bert_tokenizer, bert_model, bert_faiss_index, bert_faiss_doc_ids, query,
            index_dir=index_dir, model_type='bert', k=k, candidate_limit=candidate_limit
        )
    elif model == 'faiss_hybrid':
        results, cleaned_query = retrieve_documents_faiss(
            tfidf_vectorizer, None, hybrid_faiss_index, hybrid_faiss_doc_ids, query,
            index_dir=index_dir, model_type='hybrid', svd_model=svd_model,
            bert_tokenizer=bert_tokenizer, bert_model=bert_model, alpha=0.5, k=k, candidate_limit=candidate_limit
        )
    else:
        return {"error": "Invalid model selected"}, 400

    elapsed_time = time.time() - start_time  # حساب الوقت المستغرق

    doc_ids_only = [doc_id for doc_id, _ in results]
    docs = collection.find({'doc_id': {'$in': doc_ids_only}})
    doc_id_to_text = {doc['doc_id']: doc.get('original_text', 'Text not available') for doc in docs}

    response_results = []
    for doc_id, score in results:
        text = doc_id_to_text.get(doc_id, 'Text not available')
        response_results.append({
            'doc_id': doc_id,
            'score': float(score),
            'text': text
        })
    # إرجاع النتائج
    if request.headers.get("accept") == "application/json":
        return {
            "cleaned_query": cleaned_query,
            "results": response_results,
            "elapsed_time": elapsed_time
        }
    else:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "cleaned_query": cleaned_query,
            "results": response_results,
            "query": query,
            "model": model,
            "dataset": dataset,
            "elapsed_time": elapsed_time
        })