import logging
import time
from typing import List, Dict, Any

from backend.app.rag.ingestion.loader import FileLoader
from backend.app.rag.ingestion.cleaner import TextCleaner
from backend.app.rag.ingestion.chunker import TextChunker
from backend.app.rag.retrieval.retriever import InMemoryRetriever
from backend.app.rag.retrieval.embedder import HuggingFaceEmbedder

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.loader = FileLoader(data_path)
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=400, chunk_overlap=80)
        
        # Access existing singletons/classes
        self.retriever = InMemoryRetriever()
        self.embedder = HuggingFaceEmbedder() # Reuses existing implementation

    def run(self):
        logger.info(f"Starting ingestion from: {self.data_path}")
        start_time = time.time()
        
        total_files = 0
        total_chunks = 0
        
        # 1. Load Files
        documents = []
        for filename, content in self.loader.load_files():
            total_files += 1
            logger.info(f"Processing file: {filename}")
            
            # 2. Clean
            cleaned_content = self.cleaner.clean(content)
            
            # Infer category (simple logic)
            category = "general"
            if "spec" in filename.lower() or "spec" in cleaned_content.lower()[:100]:
                category = "specification"
            elif "comparison" in filename.lower():
                category = "comparison"
            
            base_metadata = {
                "source": filename,
                "category": category
            }
            
            # 3. Chunk
            file_chunks = self.chunker.chunk(cleaned_content, base_metadata)
            documents.extend(file_chunks)
            total_chunks += len(file_chunks)

        if not documents:
            logger.warning("No documents found or processed.")
            return

        logger.info(f"Generated {total_chunks} chunks from {total_files} files.")
        
        # 4. Embed
        logger.info("Generating embeddings...")
        texts = [doc["text"] for doc in documents]
        try:
            vectors = self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return

        # 5. Store
        logger.info("Storing to FAISS...")
        
        # Format for retriever.add_documents
        # Expected: vectors: List[List[float]], documents: List[Dict]
        # Documents payload needs to be structured as the retriever expects (usually payload + metadata)
        # Based on InMemoryRetriever._seed_dummy_data:
        # {
        #    "id": "...",
        #    "metadata": { ... },
        #    "payload": { "title": "...", "content": "..." }
        # }
        
        formatted_docs = []
        for i, doc in enumerate(documents):
            formatted_docs.append({
                "id": f"{doc['metadata']['source']}_{doc['metadata']['chunk_id']}",
                "metadata": doc['metadata'],
                "payload": {
                    "title": f"{doc['metadata']['source']} - Chunk {doc['metadata']['chunk_id']}",
                    "content": doc['text']
                }
            })
            
        self.retriever.vector_store.add(vectors, formatted_docs)
        
        elapsed = time.time() - start_time
        logger.info("==========================================")
        logger.info(f"INGESTION COMPLETE in {elapsed:.2f}s")
        logger.info(f"Files Processed: {total_files}")
        logger.info(f"Chunks Created : {total_chunks}")
        logger.info(f"Vectors Stored : {len(vectors)}")
        logger.info("==========================================")
