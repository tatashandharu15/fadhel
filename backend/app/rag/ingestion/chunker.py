from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        """
        Args:
            chunk_size: Target size in 'tokens' (approximated by words)
            chunk_overlap: Overlap size in 'tokens'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits text into chunks and assigns metadata.
        Uses simple word-based splitting as a proxy for tokens.
        """
        if not text:
            return []

        words = text.split()
        chunks = []
        
        # If text is shorter than chunk size, return as single chunk
        if len(words) <= self.chunk_size:
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_id": 0,
                "chunk_size": len(words)
            })
            return [{
                "text": text,
                "metadata": chunk_meta
            }]

        # Sliding window
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_id": len(chunks),
                "chunk_size": len(chunk_words)
            })
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_meta
            })
            
            # Stop if we reached the end
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
