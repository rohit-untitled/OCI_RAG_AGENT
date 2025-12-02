import os
import logging
import asyncio
import json
import time
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, conint
from typing import List, Optional

# ---- Internal imports ----
from app.services.document_loader import load_docx_files
from app.services.docx_extractor import extract_text_with_formatting_in_sequence
from app.services.document_chunker import chunk_documents
from app.services.chunk_service import chunk_anonymized_documents
from app.services.anonymize_service import anonymize_documents
from app.services.embedding_service import OCIEmbeddingService
from app.services.rag_service import answer_query, ai_redact_sensitive_info
from app.services.vector_store_service import insert_embeddings_from_json
from app.services.oci_downloader import download_all_from_bucket



# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_redaction_agent")

app = FastAPI(
    title="AI Redaction Agent",
    description="RAG-powered system for privacy-safe document Q&A and redaction",
    version="1.0.0",
)


def get_docs_folder() -> str:
    return os.path.join(os.path.dirname(__file__), "data", "downloads")


@app.get("/")
def root():
    return {"message": "AI Redaction Agent is running!"}

@app.get("/sync-bucket")
def sync_bucket():
    try:
        download_all_from_bucket()
        return {"message": "All documents downloaded from OCI."}
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(500, str(e))


@app.get("/load-docs")
def load_docs():
    folder = get_docs_folder()
    docs = load_docx_files(folder)

    return {
        "total_documents": len(docs),
        "documents": [
            {
                "file": doc["file_name"],
                "folder": doc["folder"],
                "path": doc["file_path"]
            }
            for doc in docs
        ]
    }



@app.get("/extract-docs")
def extract_docs():
    folder = get_docs_folder()
    docs = load_docx_files(folder)
    result = {}
    for doc in docs:
        try:
            text = extract_text_with_formatting_in_sequence(doc["file_path"])
        except Exception as e:
            text = f"Error extracting: {e}"
        result[os.path.basename(doc["file_path"])] = text
    return result


@app.get("/anonymize-docs")
def anonymize_docs():
    folder = get_docs_folder()
    return anonymize_documents(folder)

@app.get("/chunk-anonymized")
def chunk_anonymized():
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    return chunk_anonymized_documents(base_dir)


@app.post("/embed-chunks")
def embed_chunks():
    chunks_path = os.path.join(os.path.dirname(__file__), "data", "chunks", "chunks.json")
    output_path = os.path.join(os.path.dirname(__file__), "data", "chunks", "chunks_with_embeddings.json")

    embedder = OCIEmbeddingService()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    output = []
    start_time = time.time()

    for idx, ch in enumerate(chunks, start=1):
        try:
            emb = embedder.embed_text(ch["chunk"])
            ch["embedding"] = emb
            if emb is None:
                logger.warning(f"Chunk {idx} embedding failed")
        except Exception as e:
            logger.error(f"Error embedding chunk {idx}: {e}")
            ch["embedding"] = None
        output.append(ch)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return {
        "message": "Embeddings created",
        "file": os.path.basename(output_path),
        "total_chunks": len(chunks),
        "time_taken_seconds": round(time.time() - start_time, 2)
    }

@app.post("/store-embeddings")
def store_embeddings_endpoint():
    json_file = os.path.join(os.path.dirname(__file__), "data", "chunks", "chunks_with_embeddings.json")
    insert_embeddings_from_json(json_file)
    return {"status": "ok"}


class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/ask")
def ask_endpoint(payload: RAGRequest):
    """
    Query the RAG system and get LLM answer
    """
    try:
        response = answer_query(payload.query, top_k=payload.top_k)
        return response
    except Exception as e:
        logger.exception(f"Error in RAG query: {e}")
        return {"error": str(e)}