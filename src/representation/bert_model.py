from transformers import AutoTokenizer, AutoModel
import torch
from pymongo import MongoClient
import joblib
import os
import numpy as np
from src.utils.clean_text import clean_text
def build_bert_embeddings(collection_name, model_path, vector_path, batch_size=8):
    """
    بناء تضمينات BERT للنصوص الخام بدون تنظيف مسبق
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vector_path = os.path.join(project_root, vector_path)
    model_path = os.path.join(project_root, model_path)

    # تحميل نموذج BERT
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # الاتصال بـ MongoDB
    client = MongoClient('localhost', 27017)
    db = client['IR_PROJECT']
    collection = db[collection_name]

    # جلب النصوص الخام مباشرة
    docs = []
    doc_ids = []
    for doc in collection.find():
        docs.append(doc['cleaned_text_bert'])  # استخدام النص الأصلي بدلاً من النظيف
        doc_ids.append(doc['doc_id'])

    # توليد التضمينات
    embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            # Mean Pooling
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            batch_embeddings = (summed / counts).numpy()

        embeddings.append(batch_embeddings)
        print(f"Processed batch {i // batch_size + 1}/{(len(docs) // batch_size) + 1}")

    embeddings = np.vstack(embeddings)

    # حفظ النتائج
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_name, model_path)
    joblib.dump(embeddings, vector_path)
    joblib.dump(doc_ids, f"{vector_path}_doc_ids")

    print(f"✅ BERT embeddings saved for {collection_name} (raw text)")

def load_bert_embeddings(model_path, vector_path):
    """تحميل نموذج BERT والتضمينات"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)
    vector_path = os.path.join(project_root, vector_path)

    model_name = joblib.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = joblib.load(vector_path)
    doc_ids = joblib.load(f"{vector_path}_doc_ids")
    
    return tokenizer, model, embeddings, doc_ids

def mean_pooling(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts

def transform_bert_query(tokenizer, model, text):
    cleaned_text = clean_text(text, model_type="bert")
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        pooled = mean_pooling(last_hidden, inputs['attention_mask'])
        query_embedding = pooled.cpu().numpy()
    
    return query_embedding, cleaned_text