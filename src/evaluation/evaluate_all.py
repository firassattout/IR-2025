import numpy as np
import pandas as pd
import ir_datasets
import os
import joblib
import os
import sys
import random
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.representation.tfidf_model import load_tfidf_model
from src.representation.bert_model import load_bert_embeddings
from src.representation.vector_store import load_faiss_index, retrieve_documents_faiss
from src.processing_ranking.query_processing import retrieve_documents_tfidf, retrieve_documents_bert, retrieve_documents_hybrid

def load_test_data(dataset_name, sample_size=100):
    dataset = ir_datasets.load(dataset_name)
    all_queries = list(dataset.queries_iter())
    sample_size = min(sample_size, len(all_queries))
    raw_queries = random.sample(all_queries, sample_size)
    queries = [{"query_id": q.query_id, "text": q.text} for q in raw_queries]
    qrels = []
    query_ids = set(q["query_id"] for q in queries)
    for qrel in dataset.qrels_iter():
        if qrel.query_id in query_ids:
            qrels.append({
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": qrel.relevance
            })

    queries_df = pd.DataFrame(queries)
    qrels_df = pd.DataFrame(qrels)

    if queries_df.empty or qrels_df.empty:
        print(f"Warning: No queries or qrels loaded for {dataset_name}")
        return pd.DataFrame(), pd.DataFrame()

    return queries_df, qrels_df

def calculate_metrics(retrieved_docs, qrels_df, query_id, k=10):
    ranked_doc_ids = [doc_id for doc_id, _ in retrieved_docs]
    ranked_doc_ids_at_k = ranked_doc_ids[:k]
    relevant_docs = set(qrels_df[qrels_df['query_id'] == query_id]['doc_id'])
    retrieved_relevant_at_k = len([doc_id for doc_id in ranked_doc_ids_at_k if doc_id in relevant_docs])
    precision_at_k = retrieved_relevant_at_k / k if k > 0 else 0
    retrieved_relevant_total = len([doc_id for doc_id in ranked_doc_ids if doc_id in relevant_docs])
    total_relevant = len(relevant_docs)
    recall = retrieved_relevant_total / total_relevant if total_relevant > 0 else 0
    mrr = 0
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_docs:
            mrr = 1 / rank
            break
    ap = 0
    relevant_count = 0
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_docs:
            relevant_count += 1
            ap += relevant_count / rank
    ap = ap / total_relevant if total_relevant > 0 else 0
    return {
        "precision@10": precision_at_k,
        "recall": recall,
        "mrr": mrr,
        "ap": ap
    }

def evaluate_tfidf_model(dataset_name, tfidf_model_path, tfidf_vector_path, index_dir=None, sample_size=100):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfidf_model_path = os.path.join(project_root, tfidf_model_path)
    tfidf_vector_path = os.path.join(project_root, tfidf_vector_path)
    tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(tfidf_model_path, tfidf_vector_path)
    queries_df, qrels_df = load_test_data(dataset_name, sample_size)
    tfidf_results = []
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['text']
        tfidf_docs, _ = retrieve_documents_tfidf(tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids, query_text, None, index_dir)
        tfidf_metrics = calculate_metrics(tfidf_docs, qrels_df, query_id)
        tfidf_metrics['query_id'] = query_id
        tfidf_metrics['model'] = 'TF-IDF'
        tfidf_results.append(tfidf_metrics)
    results_df = pd.DataFrame(tfidf_results) if tfidf_results else pd.DataFrame()
    numeric_columns = ['precision@10', 'recall', 'mrr', 'ap']
    avg_metrics = results_df[numeric_columns].mean().to_dict() if not results_df.empty else {}
    return results_df, {'TF-IDF': avg_metrics}

def evaluate_bert_model(dataset_name, bert_model_path, bert_vector_path, index_dir=None, sample_size=100):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bert_model_path = os.path.join(project_root, bert_model_path)
    bert_vector_path = os.path.join(project_root, bert_vector_path)
    tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(bert_model_path, bert_vector_path)
    queries_df, qrels_df = load_test_data(dataset_name, sample_size)
    bert_results = []
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['text']
        bert_docs, _ = retrieve_documents_bert(tokenizer, bert_model, bert_embeddings, bert_doc_ids, query_text, index_dir)
        bert_metrics = calculate_metrics(bert_docs, qrels_df, query_id)
        bert_metrics['query_id'] = query_id
        bert_metrics['model'] = 'BERT'
        bert_results.append(bert_metrics)
    results_df = pd.DataFrame(bert_results) if bert_results else pd.DataFrame()
    numeric_columns = ['precision@10', 'recall', 'mrr', 'ap']
    avg_metrics = results_df[numeric_columns].mean().to_dict() if not results_df.empty else {}
    return results_df, {'BERT': avg_metrics}

def evaluate_hybrid_model(dataset_name, tfidf_model_path, tfidf_vector_path, svd_path, bert_model_path, bert_vector_path, hybrid_vector_path, index_dir=None, sample_size=100, alpha=0.5,k=10):
 
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfidf_model_path = os.path.join(project_root, tfidf_model_path)
    tfidf_vector_path = os.path.join(project_root, tfidf_vector_path)
    svd_path = os.path.join(project_root, svd_path)
    bert_model_path = os.path.join(project_root, bert_model_path)
    svd_model = joblib.load(svd_path)
    bert_vector_path = os.path.join(project_root, bert_vector_path)
    hybrid_vector_path = os.path.join(project_root, hybrid_vector_path)

    tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(tfidf_model_path, tfidf_vector_path)
    tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(bert_model_path, bert_vector_path)
    hybrid_vectors = joblib.load(hybrid_vector_path)
    hybrid_doc_ids = joblib.load(f"{hybrid_vector_path}_doc_ids")

    queries_df, qrels_df = load_test_data(dataset_name, sample_size)

    hybrid_results = []
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['text']
        hybrid_docs, _ = retrieve_documents_hybrid(tfidf_vectorizer, svd_model, tokenizer, bert_model, hybrid_vectors, hybrid_doc_ids, query_text, index_dir, alpha)
        hybrid_metrics = calculate_metrics(hybrid_docs, qrels_df, query_id)
        hybrid_metrics['query_id'] = query_id
        hybrid_metrics['model'] = 'Hybrid'
        hybrid_results.append(hybrid_metrics)

    results_df = pd.DataFrame(hybrid_results) if hybrid_results else pd.DataFrame()
    numeric_columns = ['precision@10', 'recall', 'mrr', 'ap']
    avg_metrics = results_df[numeric_columns].mean().to_dict() if not results_df.empty else {}
    return results_df, {'Hybrid': avg_metrics}

def evaluate_faiss_model(dataset_name,dataset_name2, index_path, vector_path, model_type, svd_path=None, bert_model_path=None, index_dir=None, sample_size=100, alpha=0.5, k=10):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(project_root, index_path)
    vector_path = os.path.join(project_root, vector_path)
    svd_path = os.path.join(project_root, svd_path) if svd_path else None
    bert_model_path = os.path.join(project_root, bert_model_path) if bert_model_path else None
    index_dir = os.path.join(project_root, index_dir) if index_dir else None
    faiss_index, doc_ids = load_faiss_index(index_path)
    if model_type == 'faiss_tfidf':
        tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(
            f"../models/tfidf/{dataset_name2}_tfidf.joblib",
            vector_path
        )
        svd_model = joblib.load(svd_path)
        vectorizer_or_tokenizer = tfidf_vectorizer
        model = None
        extra_params = {"svd_model": svd_model}

        vectorizer_or_tokenizer = tfidf_vectorizer
        model = None
        extra_params = {"svd_model": svd_model}
    elif model_type == 'faiss_bert':
        if not bert_model_path:
            raise ValueError("bert_model_path is required for FAISS BERT")
        tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
            bert_model_path, vector_path
        )
        vectorizer_or_tokenizer = tokenizer
        model = bert_model
        extra_params = {}
    elif model_type == 'faiss_hybrid':
        if not (svd_path and bert_model_path):
            raise ValueError("Both svd_path and bert_model_path are required for FAISS Hybrid")
        tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = load_tfidf_model(
            f"../models/tfidf/{dataset_name2}_tfidf.joblib",
            vector_path
        )
        svd_model = joblib.load(svd_path)
        tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
            bert_model_path, vector_path
        )
        vectorizer_or_tokenizer = tfidf_vectorizer
        model = None
        extra_params = {"svd_model": svd_model, "bert_tokenizer": tokenizer, "bert_model": bert_model, "alpha": alpha}
    else:
        raise ValueError("Invalid model_type. Choose 'faiss_tfidf', 'faiss_bert', or 'faiss_hybrid'.")

    queries_df, qrels_df = load_test_data(dataset_name, sample_size)
    if queries_df.empty or qrels_df.empty:
        return pd.DataFrame(), {model_type: {}}

    results = []
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['text']
        retrieved_docs, cleaned_query = retrieve_documents_faiss(
            vectorizer_or_tokenizer=vectorizer_or_tokenizer,
            model=model,
            vectors_or_index=faiss_index,
            doc_ids=doc_ids,
            query=query_text,
            index_dir=index_dir,
            model_type=model_type.split('_')[1],
            k=k,
            candidate_limit=100,
            **extra_params
        )
        metrics = calculate_metrics(retrieved_docs, qrels_df, query_id, k=k)
        metrics['query_id'] = query_id
        metrics['model'] = model_type
        results.append(metrics)

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    numeric_columns = ['precision@10', 'recall', 'mrr', 'ap']
    avg_metrics = results_df[numeric_columns].mean().to_dict() if not results_df.empty else {}
    return results_df, {model_type: avg_metrics}


def evaluate_all_models(dataset_name,dataset_name2, tfidf_model_path, tfidf_vector_path, svd_path, bert_model_path, bert_vector_path, hybrid_vector_path, faiss_tfidf_index_path, faiss_bert_index_path, faiss_hybrid_index_path, index_dir=None, sample_size=100, alpha=0.5, k=10):
    tfidf_results_df, tfidf_avg_metrics = evaluate_tfidf_model(dataset_name, tfidf_model_path, tfidf_vector_path, index_dir, sample_size)
    bert_results_df, bert_avg_metrics = evaluate_bert_model(dataset_name, bert_model_path, bert_vector_path, index_dir, sample_size)
    hybrid_results_df, hybrid_avg_metrics = evaluate_hybrid_model(dataset_name, tfidf_model_path, tfidf_vector_path, svd_path, bert_model_path, bert_vector_path,hybrid_vector_path, index_dir, sample_size, alpha, k)

    
    results = pd.concat([tfidf_results_df, bert_results_df, hybrid_results_df], ignore_index=True) if not (tfidf_results_df.empty and bert_results_df.empty and hybrid_results_df.empty ) else pd.DataFrame()
    avg_metrics = {
            **tfidf_avg_metrics,
            **bert_avg_metrics,
            **hybrid_avg_metrics,
    
    }
    return results, avg_metrics

def evaluate_all_with_faiss_models(dataset_name,dataset_name2, tfidf_model_path, tfidf_vector_path, svd_path, bert_model_path, bert_vector_path, hybrid_vector_path, faiss_tfidf_index_path, faiss_bert_index_path, faiss_hybrid_index_path, index_dir=None, sample_size=100, alpha=0.5, k=10):
    faiss_tfidf_results_df, faiss_tfidf_avg_metrics = evaluate_faiss_model(dataset_name,dataset_name2, faiss_tfidf_index_path, tfidf_vector_path, 'faiss_tfidf', svd_path, index_dir=index_dir, sample_size=sample_size, k=k)
    faiss_bert_results_df, faiss_bert_avg_metrics = evaluate_faiss_model(dataset_name,dataset_name2, faiss_bert_index_path, bert_vector_path, 'faiss_bert', bert_model_path=bert_model_path, index_dir=index_dir, sample_size=sample_size, k=k)
    faiss_hybrid_results_df, faiss_hybrid_avg_metrics = evaluate_faiss_model(dataset_name,dataset_name2, faiss_hybrid_index_path, hybrid_vector_path, 'faiss_hybrid', svd_path=svd_path, bert_model_path=bert_model_path, index_dir=index_dir, sample_size=sample_size, alpha=alpha, k=k)
    
    results = pd.concat([faiss_tfidf_results_df, faiss_bert_results_df, faiss_hybrid_results_df], ignore_index=True) if not ( faiss_tfidf_results_df.empty and faiss_bert_results_df.empty and faiss_hybrid_results_df.empty) else pd.DataFrame()
    avg_metrics = {

        **faiss_tfidf_avg_metrics,
            **faiss_bert_avg_metrics,
            **faiss_hybrid_avg_metrics
    }
    return results, avg_metrics
