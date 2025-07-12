from rank_bm25 import BM25Okapi
from pymongo import MongoClient
import joblib
import os
from src.utils.clean_text import clean_text
def build_bm25_model(collection_name, model_path, vector_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)
    vector_path = os.path.join(project_root, vector_path)

    client = MongoClient('localhost', 27017)
    db = client['IR_PROJECT']
    collection = db[collection_name]
    docs = []
    doc_ids = []

    for doc in collection.find():
        text = doc.get('cleaned_text_tfidf', '')
        if text:
             docs.append(text)
             doc_ids.append(doc['doc_id'])

    if not docs:
        raise ValueError(" لا توجد وثائق مناسبة لبناء نموذج BM25")

    bm25 = BM25Okapi(docs)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bm25, model_path)
    joblib.dump(doc_ids, f"{vector_path}_doc_ids")
    print(f" تم حفظ نموذج BM25 في: {model_path}")
    print(f" تم حفظ معرفات الوثائق في: {vector_path}_doc_ids")

def load_bm25_model(model_path, vector_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)
    vector_path = os.path.join(project_root, vector_path)

    bm25 = joblib.load(model_path)
    doc_ids = joblib.load(f"{vector_path}_doc_ids")

    return bm25, doc_ids

def transform_bm25_query(bm25, query_text):
    cleaned_query = clean_text(query_text)
    tokenized_query = cleaned_query.split()
    scores = bm25.get_scores(tokenized_query)
    return scores, cleaned_query