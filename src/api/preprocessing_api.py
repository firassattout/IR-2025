# src/apis/preprocessing_api.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
from src.preprocessing.data_loader import  load_and_store_data

app = FastAPI(title="Preprocessing API")


class DatasetInput(BaseModel):
    dataset_name: str
    collection_name: str


@app.post("/load_and_store")
def api_load_and_store(data: DatasetInput):
    Dataset,Number,qrels= load_and_store_data(data.dataset_name, data.collection_name)
    return {"status": "success","Dataset":Dataset,"Number":Number,"qrels":qrels}
