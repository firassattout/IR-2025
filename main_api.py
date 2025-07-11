
from fastapi import FastAPI
from src.api.preprocessing_api import app as preprocessing_app
from src.api.tfidf_api import app as tfidf_app
from src.api.bm25_api import app as bm25_app
from src.api.bert_api import app as bert_app
from src.api.hybrid_api import app as hybrid_app
from src.api.query_processing_api import app as query_processing_app
from src.api.evaluation_api import app as evaluation_app
from src.api.frontend_api import app as frontend_app
from src.api.indexing_api import app as indexing_app
from src.api.faiss_build_api import app as faiss_build_app

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/preprocess", preprocessing_app)
app.mount("/tfidf", tfidf_app)
app.mount("/bm25", bm25_app)
app.mount("/bert", bert_app)
app.mount("/hybrid", hybrid_app)
app.mount("/query_processing", query_processing_app)
app.mount("/evaluation", evaluation_app)
app.mount("/indexing", indexing_app)
app.mount("/faiss_build", faiss_build_app)
app.mount("/frontend", frontend_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
