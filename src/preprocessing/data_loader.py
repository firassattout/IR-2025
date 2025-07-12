from pymongo import MongoClient
import ir_datasets

from src.utils.clean_text import clean_text 

def load_and_store_data(dataset_name, collection_name):
    client = MongoClient('localhost', 27017)
    db = client['IR_PROJECT']
    collection = db[collection_name]
    dataset = ir_datasets.load(dataset_name)
    print("Dataset :"+ dataset_name)
    print(f"Number of documents: {dataset.docs_count()}")
    print(f"Has qrels: {dataset.has_qrels()}")

    for doc in dataset.docs_iter():
        text = doc.text if hasattr(doc, 'text') else doc.body
        collection.insert_one({
            'doc_id': doc.doc_id,
            'original_text': text,
            'cleaned_text_tfidf': clean_text(text, model_type='tfidf'),
            'cleaned_text_bert': clean_text(text, model_type='bert'),
        })

    print(f"{collection_name} documents stored: {collection.count_documents({})}")
    Dataset=dataset_name
    Number=dataset.docs_count()
    qrels=dataset.has_qrels()
    return Dataset,Number,qrels