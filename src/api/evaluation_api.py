
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.evaluation.evaluate_all import evaluate_tfidf_model, evaluate_bert_model, evaluate_hybrid_model, evaluate_all_models,evaluate_all_with_faiss_models
import os
import pandas as pd

app = FastAPI(title="Evaluation API")

class EvaluateTFIDFRequest(BaseModel):
    dataset_name: str
    tfidf_model_path: str
    tfidf_vector_path: str
    index_dir: Optional[str] = None
    sample_size: Optional[int] = 100

class EvaluateBERTRequest(BaseModel):
    dataset_name: str
    bert_model_path: str
    bert_vector_path: str
    index_dir: Optional[str] = None
    sample_size: Optional[int] = 100

class EvaluateHybridRequest(BaseModel):
    dataset_name: str
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: str
    bert_model_path: str
    bert_vector_path: str
    hybrid_vector_path: str
    index_dir: Optional[str] = None
    sample_size: Optional[int] = 100
    alpha: Optional[float] = 0.5

class EvaluateAllRequest(BaseModel):
    dataset_name: str
    dataset_name2: str
    tfidf_model_path: str
    tfidf_vector_path: str
    svd_path: str
    bert_model_path: str
    bert_vector_path: str
    hybrid_vector_path: str
    faiss_tfidf_index_path: str
    faiss_bert_index_path: str
    faiss_hybrid_index_path: str
    index_dir: Optional[str] = None
    sample_size: Optional[int] = 100
    alpha: Optional[float] = 0.5
    k: Optional[int] = 10

@app.post("/evaluate_tfidf_model")
async def api_evaluate_tfidf_model(data: EvaluateTFIDFRequest):
        results_df, avg_metrics = evaluate_tfidf_model(
            dataset_name=data.dataset_name,
            tfidf_model_path=data.tfidf_model_path,
            tfidf_vector_path=data.tfidf_vector_path,
            index_dir=data.index_dir,
            sample_size=data.sample_size
        )
        results_json = results_df.to_dict(orient="records") if not results_df.empty else []
        avg_metrics_json = {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in avg_metrics.items()
        }
        return {
            "results": results_json,
            "average_metrics": avg_metrics_json
        }


@app.post("/evaluate_bert_model")
async def api_evaluate_bert_model(data: EvaluateBERTRequest):
        results_df, avg_metrics = evaluate_bert_model(
            dataset_name=data.dataset_name,
            bert_model_path=data.bert_model_path,
            bert_vector_path=data.bert_vector_path,
            index_dir=data.index_dir,
            sample_size=data.sample_size
        )
        results_json = results_df.to_dict(orient="records") if not results_df.empty else []
        avg_metrics_json = {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in avg_metrics.items()
        }
        return {
            "results": results_json,
            "average_metrics": avg_metrics_json
        }


@app.post("/evaluate_hybrid_model")
async def api_evaluate_hybrid_model(data: EvaluateHybridRequest):
        results_df, avg_metrics = evaluate_hybrid_model(
            dataset_name=data.dataset_name,
            tfidf_model_path=data.tfidf_model_path,
            tfidf_vector_path=data.tfidf_vector_path,
            svd_path=data.svd_path,
            bert_model_path=data.bert_model_path,
            bert_vector_path=data.bert_vector_path,
            hybrid_vector_path=data.hybrid_vector_path,
            index_dir=data.index_dir,
            sample_size=data.sample_size,
            alpha=data.alpha
        )
        results_json = results_df.to_dict(orient="records") if not results_df.empty else []
        avg_metrics_json = {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in avg_metrics.items()
        }
        return {
            "results": results_json,
            "average_metrics": avg_metrics_json
        }


@app.post("/evaluate_all_models")
async def api_evaluate_all_models(data: EvaluateAllRequest):
        results_df, avg_metrics = evaluate_all_models(
            dataset_name=data.dataset_name,
            dataset_name2=data.dataset_name2,
            tfidf_model_path=data.tfidf_model_path,
            tfidf_vector_path=data.tfidf_vector_path,
            svd_path=data.svd_path,
            bert_model_path=data.bert_model_path,
            bert_vector_path=data.bert_vector_path,
            hybrid_vector_path=data.hybrid_vector_path,
            faiss_tfidf_index_path=data.faiss_tfidf_index_path,
            faiss_bert_index_path=data.faiss_bert_index_path,
            faiss_hybrid_index_path=data.faiss_hybrid_index_path,
            index_dir=data.index_dir,
            sample_size=data.sample_size,
            alpha=data.alpha,
            k=data.k
        )
        results_json = results_df.to_dict(orient="records") if not results_df.empty else []
        avg_metrics_json = {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in avg_metrics.items()
        }
        return {
            "results": results_json,
            "average_metrics": avg_metrics_json
        }

@app.post("/evaluate_all_models_with_faiss")
async def api_evaluate_all_with_faiss_models(data: EvaluateAllRequest):
        results_df, avg_metrics = evaluate_all_with_faiss_models(
            dataset_name=data.dataset_name,
            dataset_name2=data.dataset_name2,
            tfidf_model_path=data.tfidf_model_path,
            tfidf_vector_path=data.tfidf_vector_path,
            svd_path=data.svd_path,
            bert_model_path=data.bert_model_path,
            bert_vector_path=data.bert_vector_path,
            hybrid_vector_path=data.hybrid_vector_path,
            faiss_tfidf_index_path=data.faiss_tfidf_index_path,
            faiss_bert_index_path=data.faiss_bert_index_path,
            faiss_hybrid_index_path=data.faiss_hybrid_index_path,
            index_dir=data.index_dir,
            sample_size=data.sample_size,
            alpha=data.alpha,
            k=data.k
        )
        results_json = results_df.to_dict(orient="records") if not results_df.empty else []
        avg_metrics_json = {
            model: {metric: float(value) for metric, value in metrics.items()}
            for model, metrics in avg_metrics.items()
        }
        return {
            "results": results_json,
            "average_metrics": avg_metrics_json
        }

