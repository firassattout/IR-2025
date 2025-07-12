import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.representation.tfidf_model import load_tfidf_model, transform_tfidf_query
from src.representation.bert_model import load_bert_embeddings, transform_bert_query
from src.representation.hybrid_model import transform_hybrid_query
from src.indexing.index_builder import search_index  # استيراد دالة البحث في الفهرس

def retrieve_documents_tfidf(tfidf_vectorizer, tfidf_matrix, doc_ids, query, svd_model=None, index_dir=None, k=500, candidate_limit=5000):
    if index_dir:
        candidate_docs = search_index(index_dir, query, fields=["cleaned_text_tfidf"], limit=candidate_limit)
        candidate_doc_ids = [doc_id for doc_id, _ in candidate_docs]
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        candidate_indices = [doc_id_to_index[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_index]
        if len(candidate_indices) < 10:
            filtered_tfidf_matrix = tfidf_matrix
            filtered_doc_ids = doc_ids
        else:
            filtered_tfidf_matrix = tfidf_matrix[candidate_indices]
            filtered_doc_ids = [doc_ids[i] for i in candidate_indices]
    else:
        filtered_tfidf_matrix = tfidf_matrix
        filtered_doc_ids = doc_ids

    query_vec, cleaned_query = transform_tfidf_query(tfidf_vectorizer, query)
    if svd_model is not None:
      filtered_tfidf_matrix = svd_model.transform(filtered_tfidf_matrix)
      query_vec = svd_model.transform(query_vec)


    if filtered_tfidf_matrix.shape[0] == 0:
        return [], cleaned_query
    similarities = cosine_similarity(query_vec, filtered_tfidf_matrix)[0] 
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(filtered_doc_ids[idx], similarities[idx]) for idx in top_indices], cleaned_query

def retrieve_documents_bert(tokenizer, bert_model, bert_embeddings, doc_ids, query, index_dir=None, k=500, candidate_limit=5000):
    if bert_embeddings.shape[0] == 0:
        print("Error: bert_embeddings is empty")
        return [], query
    if index_dir:
        candidate_docs = search_index(index_dir, query, fields=["cleaned_text_bert"], limit=candidate_limit)
        candidate_doc_ids = [doc_id for doc_id, _ in candidate_docs]

        candidate_indices = [doc_ids.index(doc_id) for doc_id in candidate_doc_ids if doc_id in doc_ids]
        if len(candidate_indices) < 10:
            filtered_embeddings = bert_embeddings
            filtered_doc_ids = doc_ids
        else:
            filtered_embeddings = bert_embeddings[candidate_indices]
            filtered_doc_ids = [doc_ids[i] for i in candidate_indices]
    else:
        filtered_embeddings = bert_embeddings
        filtered_doc_ids = doc_ids

    query_vec,cleaned_query = transform_bert_query(tokenizer, bert_model, query)
    if filtered_embeddings.shape[0] == 0:
        return []

    similarities = cosine_similarity(query_vec, filtered_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(filtered_doc_ids[idx], similarities[idx]) for idx in top_indices],cleaned_query

def retrieve_documents_hybrid(tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, hybrid_vectors, doc_ids, query, index_dir=None, alpha=0.5, k=500, candidate_limit=5000):
    if index_dir:
        candidate_docs = search_index(index_dir, query, fields=["cleaned_text_bert","cleaned_text_tfidf"], limit=candidate_limit)
        candidate_doc_ids = [doc_id for doc_id, _ in candidate_docs]
        candidate_indices = [doc_ids.index(doc_id) for doc_id in candidate_doc_ids if doc_id in doc_ids]
        if len(candidate_indices) < 10:
            filtered_hybrid_vectors = hybrid_vectors
            filtered_doc_ids = doc_ids
        else:
            filtered_hybrid_vectors = hybrid_vectors[candidate_indices]
            filtered_doc_ids = [doc_ids[i] for i in candidate_indices]
    else:
        filtered_hybrid_vectors = hybrid_vectors
        filtered_doc_ids = doc_ids

    query_vec,cleaned_text = transform_hybrid_query(tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, query, alpha)

    similarities = cosine_similarity(query_vec, filtered_hybrid_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(filtered_doc_ids[idx], similarities[idx]) for idx in top_indices],cleaned_text
