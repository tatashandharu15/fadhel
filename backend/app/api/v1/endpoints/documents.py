from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import logging
from pydantic import BaseModel
from backend.app.rag.retrieval.pipeline import RetrievalPipeline
import io

logger = logging.getLogger(__name__)

router = APIRouter()

# Single instance for now (in-memory persistence)
# In production, this should be injected or singleton via dependency override
pipeline = RetrievalPipeline()

class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str

@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF or Text) into the knowledge base.
    """
    logger.info(f"Received file upload: {file.filename}")
    
    try:
        content = await file.read()
        text = ""
        
        if file.filename.lower().endswith(".pdf"):
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                raise HTTPException(status_code=500, detail="pypdf library not installed")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
                
        else:
            # Assume text/plain or similar
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File must be valid UTF-8 text or PDF")
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty document content")
            
        # Ingest
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "source": "upload"
        }
        
        chunks_count = await pipeline.ingest(text, metadata)
        
        return IngestResponse(
            filename=file.filename,
            chunks_added=chunks_count,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
