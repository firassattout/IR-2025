import numpy as np
import os
import joblib
from src.representation.tfidf_model import load_tfidf_model
from src.representation.bert_model import load_bert_embeddings
from src.representation.tfidf_model import load_tfidf_model, transform_tfidf_query
from src.representation.bert_model import load_bert_embeddings, transform_bert_query

def build_hybrid_model(
    tfidf_model_path,
    tfidf_vector_path,
    svd_path,
    bert_model_path,
    bert_vector_path,
    hybrid_vector_path,
    alpha=0.5,
    n_components=384
):
    """بناء تمثيل هجين باستخدام TF-IDF (مع SVD) وBERT."""

    # المسارات المطلقة
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfidf_model_path = os.path.join(project_root, tfidf_model_path)
    tfidf_vector_path = os.path.join(project_root, tfidf_vector_path)
    svd_path = os.path.join(project_root, svd_path)
    bert_model_path = os.path.join(project_root, bert_model_path)
    bert_vector_path = os.path.join(project_root, bert_vector_path)
    hybrid_vector_path = os.path.join(project_root, hybrid_vector_path)

    # تحميل بيانات TF-IDF
    tfidf_vectorizer, tfidf_matrix, svd_model, tfidf_doc_ids = load_tfidf_model(
        tfidf_model_path, tfidf_vector_path, svd_path
    )

    # تقليل الأبعاد
    print("🔄 تقليل أبعاد TF-IDF باستخدام SVD...")
    tfidf_reduced = svd_model.transform(tfidf_matrix)
    print(f"✅ شكل مصفوفة TF-IDF بعد التخفيض: {tfidf_reduced.shape}")

    # تحميل تمثيلات BERT
    tokenizer, bert_model, bert_embeddings, bert_doc_ids = load_bert_embeddings(
        bert_model_path, bert_vector_path
    )

    if len(bert_doc_ids) > len(tfidf_doc_ids) and bert_doc_ids[0] == bert_doc_ids[1]:
        print("⚠️ إزالة أول عنصر مكرر من BERT doc_ids وembeddings")
        bert_doc_ids = bert_doc_ids[1:]
        bert_embeddings = bert_embeddings[1:]

    # التأكد من تطابق الترتيب
    if tfidf_doc_ids != bert_doc_ids:
        print("⚠️ تحذير: ترتيب المعرفات غير متطابق. سيتم المتابعة لكن تأكد من الاتساق.")

    # تطبيع المتجهات
    print("🧪 تطبيع المتجهات...")
    tfidf_norm = tfidf_reduced / (np.linalg.norm(tfidf_reduced, axis=1, keepdims=True) + 1e-10)
    bert_norm = bert_embeddings / (np.linalg.norm(bert_embeddings, axis=1, keepdims=True) + 1e-10)

    # دمج المتجهات
    print("🔗 دمج المتجهات باستخدام alpha =", alpha)
    hybrid_vectors = alpha * tfidf_norm + (1 - alpha) * bert_norm

    # حفظ النتائج
    os.makedirs(os.path.dirname(hybrid_vector_path), exist_ok=True)
    joblib.dump(hybrid_vectors, hybrid_vector_path)
    joblib.dump(tfidf_doc_ids, f"{hybrid_vector_path}_doc_ids")

    print(f"✅ تم حفظ المتجهات الهجينة في: {hybrid_vector_path}")
    
def transform_hybrid_query(tfidf_vectorizer, svd_model, bert_tokenizer, bert_model, query, alpha=0.5):

    tfidf_query_vec, cleaned_text = transform_tfidf_query(tfidf_vectorizer, query)
    tfidf_query_vec_dense = tfidf_query_vec.toarray()  # تحويل sparse إلى dense
    tfidf_query_reduced = svd_model.transform(tfidf_query_vec_dense)  # تقليل الأبعاد
    tfidf_query_norm = tfidf_query_reduced / (np.linalg.norm(tfidf_query_reduced, axis=1, keepdims=True) + 1e-10)


    # تحويل الاستعلام باستخدام BERT
    bert_query_vec,cleaned_query = transform_bert_query(bert_tokenizer, bert_model, query)
    bert_query_norm = bert_query_vec / (np.linalg.norm(bert_query_vec, axis=1, keepdims=True) + 1e-10)

    # دمج المتجهات
    hybrid_query_vec = alpha * tfidf_query_norm + (1 - alpha) * bert_query_norm

    return hybrid_query_vec,cleaned_text

