import faiss
import numpy as np
import os
import joblib
from src.representation.tfidf_model import load_tfidf_model, transform_tfidf_query
from src.representation.bert_model import load_bert_embeddings, transform_bert_query
from src.representation.hybrid_model import transform_hybrid_query

def build_faiss_index(vectors, doc_ids, index_path):
    vectors = vectors.astype(np.float32)
    dimension = vectors.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    joblib.dump(doc_ids, f"{index_path}_doc_ids")
    print(f"✅ FAISS index saved at: {index_path}")

def load_faiss_index(index_path):

    index = faiss.read_index(index_path)
    doc_ids = joblib.load(f"{index_path}_doc_ids")
    return index, doc_ids

def retrieve_documents_faiss(vectorizer_or_tokenizer, model, vectors_or_index, doc_ids, query,
                              index_dir=None, model_type='tfidf', svd_model=None,
                              bert_tokenizer=None, bert_model=None, alpha=0.5, k=10, candidate_limit=100):
    if model_type == 'tfidf':
        query_vec, cleaned_query = transform_tfidf_query(vectorizer_or_tokenizer, query)
        query_vec = query_vec.toarray()
        if svd_model is not None:
            query_vec = svd_model.transform(query_vec)
    elif model_type == 'bert':
        query_vec, cleaned_query = transform_bert_query(vectorizer_or_tokenizer, model, query)
    elif model_type == 'hybrid':
        query_vec, cleaned_query = transform_hybrid_query(vectorizer_or_tokenizer, svd_model, bert_tokenizer, bert_model, query, alpha)
    else:
        raise ValueError("Invalid model_type")

    query_vec = query_vec.astype(np.float32)

    distances, indices = vectors_or_index.search(query_vec, k)
    scores = 1 / (1 + distances[0]) 

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(doc_ids):
            results.append((doc_ids[idx], scores[i]))

    return results, cleaned_query