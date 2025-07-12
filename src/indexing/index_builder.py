from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
from pymongo import MongoClient
import os
from src.utils.clean_text import clean_text

def build_index(collection_name, index_dir, fields=["cleaned_text"]):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(project_root, index_dir)
    

    os.makedirs(index_dir, exist_ok=True)
    
    schema = Schema(
        doc_id=ID(stored=True),
        **{field: TEXT(stored=True) for field in fields}  
    )
    
    index = create_in(index_dir, schema)
    writer = index.writer()
    
    client = MongoClient('localhost', 27017)
    db = client['IR_PROJECT']
    collection = db[collection_name]
    
    for doc in collection.find():
        doc_data = {"doc_id": str(doc["doc_id"])}
        for field in fields:
            if field in doc:
             doc_data[field] = doc[field]
        writer.add_document(**doc_data)
    
    writer.commit()
    print(f"✅ Index built for {collection_name} in {index_dir}")

def search_index(index_dir, query_text, fields=["cleaned_text"], limit=10):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(project_root, index_dir)
    
    index = open_dir(index_dir)
    with index.searcher() as searcher:
        query_parser = MultifieldParser(fields, index.schema)
        
        query_input = clean_text(query_text) if fields == ["cleaned_text"] else query_text
        
        query = query_parser.parse(query_input)
        results = searcher.search(query, limit=limit)
        return [(result["doc_id"], {field: result[field] for field in fields}) for result in results]
