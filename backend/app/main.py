from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.app.api.v1.endpoints import chat, documents
from backend.app.llm.factory import LLMFactory
from backend.app.rag.retrieval.embedder import HuggingFaceEmbedder
from backend.app.rag.ingestion.pipeline import IngestionPipeline
import logging
import asyncio
import os
import sys

# Ensure logs are flushed immediately
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload Models
    logger.info("🚀 System Startup: Preloading AI Models...")
    
    # 1. Preload Embedding Model
    logger.info("📦 Preloading Embedding Model...")
    # Run in thread pool to not block startup completely, though startup is blocking by nature
    embedder = HuggingFaceEmbedder()
    # Trigger download/load
    await asyncio.to_thread(embedder._ensure_model)
    
    # 2. Ingest Data (if available)
    data_path = "data"
    if os.path.exists(data_path):
        logger.info(f"📂 Found data directory: {data_path}. Starting ingestion...")
        try:
            pipeline = IngestionPipeline(data_path)
            # Run in thread to avoid blocking loop
            await asyncio.to_thread(pipeline.run)
            logger.info("✅ Data ingestion completed.")
        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
    else:
        logger.warning(f"⚠️ No data directory found at {data_path}. Skipping ingestion.")

    # 3. Preload LLM (Optional but recommended for user experience)
    # Now enabled to prevent healthcheck timeout during first query
    default_model_id = os.getenv("DEFAULT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    logger.info(f"📦 Preloading LLM {default_model_id}...")
    try:
        provider = LLMFactory.get_provider(default_model_id)
        await provider._ensure_model_loaded()
        logger.info(f"✅ LLM {default_model_id} preloaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️ LLM preloading failed (will retry on first request): {e}")
    
    logger.info("✅ System Ready. Models will load on demand.")
    yield
    # Shutdown logic if any
    logger.info("🛑 System Shutdown")

app = FastAPI(
    title="Enterprise AI RAG System",
    description="API-first Modular AI Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Register Routers
app.include_router(chat.router, prefix="/v1/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/v1/documents", tags=["Documents"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "AI System Operational. Use /docs for API contract."}
