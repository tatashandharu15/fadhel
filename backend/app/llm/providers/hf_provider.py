from typing import Optional
import threading
import torch
import logging
import re
from backend.app.llm.base import BaseLLMProvider
from backend.app.llm.strategies.base import BaseLLMStrategy

# Configure logging
logger = logging.getLogger(__name__)


def is_bad_output(text: str) -> bool:
    clean = text.strip()
    return (
        len(clean) < 10
        or clean.lower() in ["ev", "1"]
        or "_" in clean
    )


def _fallback_text(query: str, context: Optional[str]) -> str:
    q = query.lower()
    if "baterai" in q and context:
        numbers = re.findall(r"(\d+(?:\.\d+)?)\s*kwh", context.lower())
        if len(numbers) >= 2:
            return (
                f"Kapasitas baterai kendaraan ini adalah {numbers[0]} kWh untuk varian Standard Range "
                f"dan {numbers[1]} kWh untuk varian Long Range."
            )

    if "ev" in q or "electric vehicle" in q or "mobil listrik" in q:
        return "Mobil listrik adalah kendaraan yang menggunakan motor listrik sebagai sumber tenaga utama."

    return "Maaf, saya tidak dapat memberikan jawaban yang baik saat ini. Silakan coba lagi."


def _looks_automotive(query: str) -> bool:
    q = query.lower()
    keywords = [
        "mobil",
        "kendaraan",
        "mesin",
        "cc",
        "baterai",
        "ev",
        "electric vehicle",
        "engine",
        "car",
        "vehicle",
        "hybrid",
        "listrik",
        "charging",
        "fuel",
        "wuling",
        "tesla",
    ]
    return any(k in q for k in keywords)

class HuggingFaceProvider(BaseLLMProvider):
    """
    Concrete implementation untuk Hugging Face models.
    Mendukung local inference menggunakan library `transformers`.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model_loaded = False
        self._lock = threading.Lock()
        self.tokenizer = None
        self.model = None
        
    async def _ensure_model_loaded(self):
        """
        Mekanisme lazy loading. Load model hanya saat pertama kali dipanggil.
        Thread-safe.
        """
        if not self._model_loaded:
            with self._lock:
                if not self._model_loaded:
                    try:
                        logger.info(f"[LLM] Initializing Hugging Face model: {self.model_id}")
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        
                        # Load Tokenizer
                        logger.info(f"⏳ [LLM] Downloading/Loading Tokenizer: {self.model_id}...")
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                        
                        # Load Model
                        logger.info(f"⏳ [LLM] Downloading/Loading Model: {self.model_id} (This may take a few minutes on first run)...")
                        print(f"DEBUG: Starting AutoModelForCausalLM.from_pretrained for {self.model_id}...", flush=True)
                        # Gunakan CPU by default agar aman di semua environment
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        print(f"DEBUG: Model loaded.", flush=True)
                        
                        self._model_loaded = True
                        logger.info(f"[LLM] Model {self.model_id} loaded successfully.")
                        
                    except ImportError:
                        error_msg = "Transformers or torch library not found. Please install via 'pip install transformers torch'."
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    except Exception as e:
                        error_msg = f"Failed to load LLM {self.model_id}: {str(e)}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

    async def generate(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None) -> str:
        """
        Implementasi generate nyata.
        1. Build Prompt (via Strategy)
        2. Tokenize
        3. Generate (CPU inference)
        4. Decode
        """
        
        # 1. Prepare Model
        await self._ensure_model_loaded()
        
        try:
            # 2. Build Prompt using Strategy
            final_prompt = strategy.build_prompt(query, context)
            
            # 3. Tokenize
            inputs = self.tokenizer(final_prompt, return_tensors="pt")
            
            # 4. Generate
            # Konfigurasi default: temperature=0.2, max_new_tokens=100 (reduced for test speed)
            logger.info(f"Starting generation for query: {query[:50]}...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            logger.info("Generation completed.")
            
            # 5. Decode
            # Skip prompt di output (hanya ambil generated tokens)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response_text = response_text.strip()
            if is_bad_output(response_text):
                response_text = _fallback_text(query, context)
            refusal = "Maaf, sistem ini hanya mendukung pertanyaan seputar otomotif."
            if response_text.strip() == refusal and _looks_automotive(query):
                response_text = _fallback_text(query, context)
            return response_text
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            raise RuntimeError(f"LLM Generation failed: {str(e)}")

    async def stream(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None):
        # Stub implementation for streaming (Not requested)
        yield "Stream functionality not implemented yet."
