from typing import List, Dict, Any
from backend.app.rag.retrieval.config import ChunkingConfig

class TextChunker:
    """
    Bertanggung jawab memecah dokumen panjang menjadi potongan kecil (chunks).
    """
    
    def __init__(self, config: ChunkingConfig = ChunkingConfig()):
        self.config = config

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Input: Text mentah, Metadata dokumen induk.
        Output: List of dict (chunk_text, chunk_metadata).
        """
        chunks = []
        
        # STUB: Implementasi simple split dulu
        # Di production, gunakan langchain.text_splitter.RecursiveCharacterTextSplitter
        
        words = text.split()
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Propagasi metadata + tambah info chunk
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = len(chunks)
            chunk_meta["chunk_size"] = len(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })
            
        return chunks
