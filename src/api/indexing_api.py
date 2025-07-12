
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.indexing.index_builder import build_index, search_index
import os

app = FastAPI(title="Indexing API")

class BuildIndexRequest(BaseModel):
    collection_name: str
    index_dir: str
    fields: List[str] = ["cleaned_text"]

class SearchIndexRequest(BaseModel):
    index_dir: str
    query_text: str
    fields: List[str] = ["cleaned_text"]
    limit: Optional[int] = 10

@app.post("/build_index")
async def api_build_index(data: BuildIndexRequest):
    build_index(data.collection_name, data.index_dir, data.fields)
    return {"status": f"Index built successfully for {data.collection_name} in {data.index_dir}"}


@app.post("/search_index")
async def api_search_index(data: SearchIndexRequest):
    results = search_index(data.index_dir, data.query_text, data.fields, data.limit)
    results_response = [
        {"doc_id": doc_id, "content": content} for doc_id, content in results
    ]
    return {
        "query_text": data.query_text,
        "results": results_response,
        "result_count": len(results_response)
    }
