import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Importing sentence_transformers...")
try:
    from sentence_transformers import SentenceTransformer
    logger.info("Imported successfully.")
except Exception as e:
    logger.error(f"Failed to import: {e}")
    sys.exit(1)

logger.info("Loading model...")
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Model loaded.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

logger.info("Encoding text...")
emb = model.encode("This is a test.")
logger.info(f"Encoded shape: {emb.shape}")
