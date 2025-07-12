from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pymongo import MongoClient
import joblib
import os

from src.utils.clean_text import clean_text

def build_tfidf_model(collection_name, model_path, vector_path, svd_path=None, n_components=100):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)
    vector_path = os.path.join(project_root, vector_path)
    if svd_path:
        svd_path = os.path.join(project_root, svd_path)

    client = MongoClient('localhost', 27017)
    db = client['IR_PROJECT']
    collection = db[collection_name]

    docs = []
    doc_ids = []
    for doc in collection.find():
        docs.append(doc['cleaned_text_tfidf'])
        doc_ids.append(doc['doc_id'])

    vectorizer = TfidfVectorizer(max_features=50000, preprocessor=clean_text)
    tfidf_matrix = vectorizer.fit_transform(docs)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(vectorizer, model_path)
    joblib.dump(tfidf_matrix, vector_path)
    joblib.dump(doc_ids, f"{vector_path}_doc_ids")
    print(f"✅ تم حفظ نموذج TF-IDF والمتجهات.")

    # تدريب وحفظ SVD إذا طُلب
    if svd_path:
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(tfidf_matrix)
        joblib.dump(svd, svd_path)
        print(f"✅ تم حفظ نموذج SVD في: {svd_path}")

def load_tfidf_model(model_path, vector_path, svd_path=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)
    vector_path = os.path.join(project_root, vector_path)
    if svd_path:
        svd_path = os.path.join(project_root, svd_path)

    vectorizer = joblib.load(model_path)
    tfidf_matrix = joblib.load(vector_path)
    doc_ids = joblib.load(f"{vector_path}_doc_ids")
    svd = joblib.load(svd_path) if svd_path and os.path.exists(svd_path) else None

    return (vectorizer, tfidf_matrix, svd, doc_ids) if svd else (vectorizer, tfidf_matrix, doc_ids)

def transform_tfidf_query(vectorizer, text):
    cleaned_text = clean_text(text)
    query_vec = vectorizer.transform([cleaned_text])
    return query_vec, cleaned_text

