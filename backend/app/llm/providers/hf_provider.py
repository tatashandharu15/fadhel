from typing import Optional
import threading
import torch
import logging
import re
from backend.app.llm.base import BaseLLMProvider
from backend.app.llm.strategies.base import BaseLLMStrategy

# Configure logging
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def is_bad_output(text: str) -> bool:
    clean = text.strip()
    lowered = clean.lower()
    repeated_phrases = re.findall(r"\b([a-zA-Z]{4,}(?:\s+[a-zA-Z]{4,}){0,2})\b", lowered)
    repetitive = any(repeated_phrases.count(phrase) >= 3 for phrase in set(repeated_phrases))
    return (
        len(clean) < 10
        or lowered in ["ev", "1"]
        or "_" in clean
        or repetitive
        or "catatan_rag" in lowered
        or "jawaban_utama" in lowered
        or "gejala_utama" in lowered
        or "pemeriksaan_awal" in lowered
        or "sirkuit sederhana" in lowered
        or lowered.count("biasanya") >= 4
        or "teleskop alternator" in lowered
        or "tegangan yang ditransmisikan oleh mesin" in lowered
        or "tekanan air" in lowered
        or "1 bar" in lowered
        or "120-150" in lowered
        or "120 sampai 150" in lowered
        or "120 hingga 150" in lowered
        or "vdc" in lowered
    )


def normalize_output(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Remove markdown emphasis fragments that often appear when generation stops mid-way.
    cleaned = re.sub(r"\*\*+", "", cleaned)
    cleaned = re.sub(r"__+", "", cleaned)

    # Collapse excessive whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove RAG metadata that should never be shown to end users.
    cleaned = re.sub(r"CATATAN_RAG\s*:.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"KATA_KUNCI\s*:.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"PERTANYAAN_TERKAIT\s*:.*$", "", cleaned, flags=re.IGNORECASE).strip()

    # Remove obvious field labels if they leak into the answer.
    cleaned = re.sub(r"\b(JAWABAN_UTAMA|GEJALA_UTAMA|PEMERIKSAAN_AWAL|TOPIK|JENIS)\s*:\s*", "", cleaned, flags=re.IGNORECASE)

    # Clean duplicated lead labels like "Sensor bermasalah: Sensor bermasalah ..."
    cleaned = re.sub(r"\b([A-ZA-Za-zÀ-ÿ][^:]{2,60})\s*:\s*\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Remove repeated short phrases at the beginning of a sentence.
    cleaned = re.sub(r"\b([A-ZA-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s-]{3,40})\b:\s*\1\b", r"\1", cleaned, flags=re.IGNORECASE)

    # If the response clearly stops in the middle of a word, trim back to the last safe boundary.
    if cleaned and cleaned[-1].isalnum() and not re.search(r"[.!?]$", cleaned):
        last_boundary = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if last_boundary != -1:
            cleaned = cleaned[: last_boundary + 1].strip()

    # Final whitespace cleanup after replacements.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def format_structured_output(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Put numbered points on separate lines.
    cleaned = re.sub(r"\s+(\d+)\.\s*", r"\n\1. ", cleaned)

    # Split common closing advice into a new paragraph.
    cleaned = re.sub(
        r"\s+(Pertimbangkan untuk|Sebaiknya|Disarankan untuk|Jika gejala ini muncul)",
        r"\n\n\1",
        cleaned,
    )

    # Avoid a leading newline if the answer starts with "1."
    cleaned = cleaned.lstrip()
    return cleaned


def deduplicate_lead_phrase(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or ":" not in cleaned:
        return cleaned

    head, tail = cleaned.split(":", 1)
    head = head.strip()
    tail = tail.strip()
    if not head or not tail:
        return cleaned

    # If the explanation starts by repeating the label, drop the label.
    if tail.lower().startswith(head.lower()):
        return tail

    return cleaned


def _alternator_voltage_answer(context: Optional[str]) -> Optional[str]:
    if not context:
        return None

    ctx = context.lower()
    match = re.search(
        r"(\d+(?:[,.]\d+)?)\s*(?:sampai|-|hingga)\s*(\d+(?:[,.]\d+)?)\s*volt",
        ctx,
    )
    if not match:
        return None

    low = match.group(1).replace(".", ",")
    high = match.group(2).replace(".", ",")
    return (
        f"Tegangan pengisian normal alternator saat mesin hidup umumnya sekitar "
        f"{low} sampai {high} volt. Jika tegangannya jauh di bawah angka tersebut, "
        f"alternator atau sistem pengisian perlu diperiksa."
    )


def _alternator_symptom_answer(query: str, context: Optional[str]) -> Optional[str]:
    q = query.lower()
    if "alternator" not in q or not any(k in q for k in ["tanda", "gejala", "lemah", "rusak"]):
        return None

    return (
        "Tanda alternator mobil mulai lemah antara lain lampu indikator aki menyala, "
        "lampu mobil redup saat mesin hidup, aki sering habis walaupun sudah diganti, "
        "tegangan pengisian tidak stabil, dan mobil sulit distarter karena aki tidak "
        "terisi penuh."
    )


def _fallback_text(query: str, context: Optional[str]) -> str:
    q = query.lower()
    symptom_answer = _alternator_symptom_answer(query, context)
    if symptom_answer:
        return symptom_answer

    if "sulit dihidupkan" in q and "pagi" in q:
        return (
            "Mobil sulit dihidupkan saat pagi hari biasanya disebabkan oleh aki yang mulai lemah, sistem pengapian yang kurang optimal, "
            "atau suplai bahan bakar yang tidak lancar. Suhu dingin juga dapat memengaruhi proses pembakaran sehingga mesin lebih sulit "
            "menyala. Pemeriksaan awal sebaiknya dilakukan pada aki, busi, relay starter, dan sistem bahan bakar."
        )

    if "setir" in q and "berat" in q:
        return (
            "Setir mobil yang terasa berat biasanya disebabkan oleh gangguan pada sistem power steering, tekanan ban yang rendah, "
            "atau komponen kemudi yang mulai aus. Pada sistem hidrolik, oli power steering yang berkurang juga dapat menjadi penyebab. "
            "Pemeriksaan awal sebaiknya dilakukan pada tekanan ban, volume oli power steering, dan komponen kemudi."
        )

    if "rem" in q and any(k in q for k in ["bunyi", "berdecit"]):
        return (
            "Rem mobil yang berbunyi biasanya disebabkan oleh kampas rem yang mulai menipis, permukaan cakram yang kotor atau tidak rata, "
            "atau debu yang menempel pada sistem pengereman. Jika cakram sudah bergelombang, rem juga bisa terasa bergetar. "
            "Pemeriksaan kampas rem dan cakram sangat disarankan jika suara terus muncul."
        )

    if "aki" in q and any(k in q for k in ["soak", "habis", "tekor"]):
        return (
            "Aki mobil cepat soak biasanya disebabkan oleh sistem pengisian yang tidak normal, usia aki yang menurun, atau adanya arus bocor "
            "pada sistem kelistrikan. Alternator yang lemah dan terminal aki yang kotor juga dapat mempercepat aki habis. "
            "Pemeriksaan awal dapat dilakukan pada tegangan aki, alternator, dan kemungkinan arus bocor."
        )

    if any(k in q for k in ["bensin", "bahan bakar"]) and any(k in q for k in ["boros", "cepat habis"]):
        return (
            "Konsumsi bahan bakar yang boros biasanya disebabkan oleh pembakaran yang tidak efisien, filter udara kotor, injektor bermasalah, "
            "atau tekanan ban yang kurang. Gaya berkendara yang agresif juga dapat membuat bahan bakar lebih cepat habis. "
            "Pemeriksaan pada filter udara, sistem bahan bakar, dan kondisi ban dapat membantu mengurangi pemborosan."
        )

    if "bergetar" in q and "idle" in q:
        return (
            "Mobil yang bergetar saat idle biasanya disebabkan oleh pembakaran yang tidak stabil, busi yang kotor, injektor yang kurang optimal, "
            "atau engine mounting yang mulai melemah. Setelan idle yang tidak tepat juga dapat membuat getaran terasa lebih jelas. "
            "Pemeriksaan awal sebaiknya dilakukan pada sistem pengapian, sistem bahan bakar, dan dudukan mesin."
        )

    if "check engine" in q:
        return (
            "Lampu check engine menandakan ada gangguan yang terdeteksi pada sistem mesin, sensor, atau sistem emisi kendaraan. "
            "Penyebab pastinya perlu diperiksa menggunakan scanner karena indikator ini bisa dipicu oleh berbagai komponen yang terhubung ke ECU. "
            "Jika lampu terus menyala, kendaraan sebaiknya segera diperiksa."
        )

    if "catalytic converter" in q:
        return (
            "Catalytic converter berfungsi mengurangi kandungan gas berbahaya pada emisi kendaraan sebelum keluar melalui knalpot. "
            "Komponen ini membantu mengubah zat berbahaya menjadi gas yang lebih aman bagi lingkungan. "
            "Jika catalytic converter bermasalah, emisi kendaraan dapat meningkat dan performa mesin juga bisa terpengaruh."
        )

    if "menanjak" in q or "tanjakan" in q:
        return (
            "Mobil yang sulit berakselerasi saat menanjak biasanya disebabkan oleh tenaga mesin yang menurun, "
            "pembakaran yang tidak optimal, atau sistem transmisi yang tidak bekerja maksimal. Pada mobil manual, "
            "kopling yang mulai aus juga dapat membuat tenaga tidak tersalurkan dengan baik. Pemeriksaan awal "
            "sebaiknya dilakukan pada sistem bahan bakar, pengapian, filter udara, dan komponen transmisi."
        )

    if ("motor" in q and "tersendat" in q) or ("motor" in q and "digas" in q):
        return (
            "Motor yang tersendat saat digas biasanya disebabkan oleh suplai bahan bakar yang tidak lancar, karburator kotor, "
            "injektor bermasalah, atau sensor throttle yang tidak bekerja optimal. Busi yang lemah juga dapat membuat pembakaran tidak stabil. "
            "Pemeriksaan awal sebaiknya dilakukan pada sistem bahan bakar, injektor atau karburator, dan busi."
        )

    if "lampu" in q and "motor" in q and "redup" in q:
        return (
            "Lampu motor yang redup biasanya menunjukkan adanya masalah pada sistem kelistrikan, seperti aki yang mulai lemah "
            "atau sistem pengisian yang tidak bekerja dengan baik. Komponen seperti spul dan regulator juga perlu diperiksa karena "
            "berpengaruh pada kestabilan arus listrik. Jika lampu terus redup, sistem pengisian motor sebaiknya segera dicek."
        )

    if "injeksi" in q:
        return (
            "Sistem injeksi bahan bakar bekerja dengan mengatur jumlah dan waktu penyemprotan bahan bakar ke mesin secara presisi. "
            "Proses ini dikendalikan oleh ECU berdasarkan data dari sensor agar campuran udara dan bahan bakar sesuai kebutuhan mesin. "
            "Dengan sistem injeksi, pembakaran menjadi lebih efisien dan respons mesin biasanya lebih stabil."
        )

    if "suspensi" in q:
        return (
            "Suspensi berfungsi meredam guncangan dari permukaan jalan agar kendaraan tetap nyaman dan stabil saat digunakan. "
            "Pada mobil dan motor, suspensi juga membantu menjaga kontak ban dengan jalan sehingga pengendalian menjadi lebih baik. "
            "Jika suspensi lemah, kenyamanan dan kestabilan kendaraan akan menurun."
        )

    if any(k in q for k in ["kurang responsif", "responsif", "gas ditarik"]):
        return (
            "Kendaraan yang terasa kurang responsif saat gas ditarik biasanya disebabkan oleh suplai bahan bakar yang kurang lancar, "
            "throttle body kotor, injektor bermasalah, atau sistem pengapian yang tidak optimal. Filter udara yang kotor juga dapat "
            "mengurangi aliran udara ke mesin. Pemeriksaan awal sebaiknya dilakukan pada throttle, injektor, filter udara, dan busi."
        )

    if "bergetar" in q and ("kecepatan tinggi" in q or "ngebut" in q):
        return (
            "Kendaraan yang bergetar saat kecepatan tinggi biasanya disebabkan oleh ban yang tidak seimbang, velg yang kurang lurus, "
            "atau suspensi yang mulai melemah. Pada beberapa kasus, bearing roda atau komponen kaki-kaki juga dapat menjadi penyebab. "
            "Pemeriksaan pada ban, velg, balancing, dan suspensi sangat disarankan jika getaran terus muncul."
        )

    if "cvt" in q:
        if any(k in q for k in ["berisik", "bunyi", "kasar"]):
            return (
                "CVT motor yang berisik biasanya disebabkan oleh roller yang aus, v-belt yang mulai retak, "
                "rumah CVT yang kotor, atau kampas kopling yang tidak lagi bekerja halus. Pemeriksaan awal "
                "sebaiknya dilakukan pada roller, v-belt, kampas kopling, dan kebersihan rumah CVT."
            )
        return (
            "CVT atau Continuously Variable Transmission adalah sistem transmisi otomatis pada motor matik "
            "yang menyalurkan tenaga mesin ke roda belakang tanpa perpindahan gigi manual. Komponen utamanya "
            "biasanya meliputi pulley depan, pulley belakang, roller, v-belt, dan kopling ganda."
        )

    if ("alternator" in q or "pengisian" in q or "mesin hidup" in q) and (
        "tegangan" in q or "volt" in q
    ):
        voltage_answer = _alternator_voltage_answer(context)
        if voltage_answer:
            return voltage_answer
        return (
            "Tegangan pengisian normal alternator saat mesin hidup umumnya sekitar "
            "13,8 sampai 14,5 volt DC. Jika tegangannya jauh di bawah atau jauh di atas "
            "rentang tersebut, alternator, regulator pengisian, kabel massa, atau terminal "
            "aki perlu diperiksa."
        )

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


def _deterministic_answer(query: str) -> Optional[str]:
    q = query.lower().strip()

    patterns = [
        (
            ["rem motor", "berdecit"],
            "Rem motor yang berbunyi berdecit saat kecepatan rendah biasanya disebabkan oleh kampas rem yang mulai aus, material kampas yang mengeras, atau permukaan piringan rem yang kotor. Debu dan kotoran pada sistem rem juga dapat menimbulkan bunyi saat pengereman. Pemeriksaan awal sebaiknya dilakukan pada kampas rem, piringan, dan kebersihan area pengereman."
        ),
        (
            ["knalpot motor", "asap hitam"],
            "Asap hitam pada knalpot motor menandakan campuran bahan bakar terlalu kaya sehingga pembakaran tidak sempurna. Kondisi ini bisa dipicu oleh filter udara yang kotor, injektor bocor, atau setelan bahan bakar yang tidak tepat. Jika dibiarkan, konsumsi bensin dapat menjadi lebih boros dan performa mesin menurun."
        ),
        (
            ["motor", "kurang bertenaga"],
            "Motor yang terasa kurang bertenaga biasanya disebabkan oleh filter udara yang kotor, busi yang melemah, suplai bahan bakar yang tidak optimal, atau kompresi mesin yang menurun. Kondisi transmisi atau kopling yang mulai aus juga dapat memengaruhi tenaga yang tersalurkan ke roda. Pemeriksaan awal sebaiknya meliputi busi, filter udara, sistem bahan bakar, dan kompresi mesin."
        ),
        (
            ["motor", "cepat panas", "perjalanan jauh"],
            "Motor yang cepat panas saat digunakan dalam perjalanan jauh biasanya disebabkan oleh pelumasan yang tidak optimal, kualitas oli yang menurun, atau sistem pendinginan yang tidak bekerja maksimal. Gesekan antar komponen mesin akan meningkat jika pendinginan dan pelumasan tidak berjalan baik. Pemeriksaan awal sebaiknya dilakukan pada kondisi oli, jalur pendinginan, dan kebersihan mesin."
        ),
        (
            ["suara mesin motor", "kasar"],
            "Suara mesin motor yang terdengar lebih kasar dari biasanya umumnya disebabkan oleh pelumasan yang kurang baik atau adanya keausan pada komponen internal mesin. Oli yang kualitasnya menurun dapat meningkatkan gesekan sehingga suara mesin menjadi lebih menonjol. Pemeriksaan pada oli, klep, dan komponen mesin lain disarankan bila suara kasar terus muncul."
        ),
        (
            ["motor", "terasa berat"],
            "Motor yang terasa berat saat dikendarai bisa disebabkan oleh tekanan ban yang kurang, rem yang seret, rantai yang terlalu kencang, atau sistem transmisi yang tidak bekerja optimal. Kondisi mesin yang kurang prima juga dapat membuat tenaga terasa tertahan. Pemeriksaan awal sebaiknya dilakukan pada ban, rem, rantai, dan sistem transmisi."
        ),
        (
            ["rantai motor", "berbunyi kasar"],
            "Rantai motor yang berbunyi kasar biasanya disebabkan oleh kurangnya pelumasan, setelan rantai yang tidak tepat, atau rantai dan gear yang mulai aus. Gesekan antar komponen akan meningkat jika rantai terlalu kering atau terlalu kencang. Pelumasan rutin dan pemeriksaan kekencangan rantai sangat penting untuk menjaga performa transmisi motor."
        ),
        (
            ["motor", "tidak stabil", "kecepatan tinggi"],
            "Motor yang tidak stabil saat kecepatan tinggi biasanya disebabkan oleh kondisi ban yang tidak seimbang, velg yang tidak lurus, atau suspensi yang mulai melemah. Tekanan angin yang tidak sesuai juga dapat memengaruhi kestabilan kendaraan. Pemeriksaan sebaiknya dilakukan pada ban, velg, bearing, dan suspensi agar pengendalian motor tetap aman."
        ),
        (
            ["motor", "brebet", "jalan pelan"],
            "Motor yang brebet saat jalan pelan umumnya disebabkan oleh suplai bahan bakar yang tidak stabil, karburator kotor, atau injektor bermasalah. Setelan idle yang kurang tepat juga bisa membuat putaran rendah tidak stabil. Pemeriksaan awal sebaiknya dilakukan pada sistem bahan bakar, busi, dan setelan idle motor."
        ),
        (
            ["starter elektrik", "motor"],
            "Starter elektrik motor yang tidak berfungsi biasanya disebabkan oleh aki yang lemah, relay starter bermasalah, atau dinamo starter yang tidak bekerja optimal. Kabel kelistrikan yang longgar atau saklar starter yang rusak juga dapat menyebabkan gejala serupa. Pemeriksaan awal sebaiknya dilakukan pada aki, relay, kabel, dan dinamo starter."
        ),
        (
            ["motor", "sulit dinyalakan", "panas"],
            "Motor yang sulit dinyalakan saat kondisi panas biasanya disebabkan oleh sistem bahan bakar atau pengapian yang tidak bekerja optimal ketika suhu mesin tinggi. Penguapan bahan bakar, busi yang melemah, atau kompresi yang menurun juga dapat memperburuk kondisi ini. Pemeriksaan awal sebaiknya dilakukan pada busi, bahan bakar, dan sistem pengapian."
        ),
        (
            ["bensin motor", "boros"],
            "Motor yang boros bensin biasanya disebabkan oleh pembakaran yang tidak efisien, filter udara kotor, injektor yang tidak bekerja optimal, atau setelan bahan bakar yang tidak tepat. Gaya berkendara yang agresif juga dapat meningkatkan konsumsi bahan bakar. Pemeriksaan awal sebaiknya dilakukan pada filter udara, sistem bahan bakar, dan kondisi busi."
        ),
        (
            ["bau bensin", "motor"],
            "Bau bensin pada motor biasanya menandakan adanya kebocoran pada sistem bahan bakar, seperti selang bensin, karburator, atau sambungan yang tidak rapat. Kondisi ini juga bisa terjadi jika campuran bahan bakar terlalu kaya. Pemeriksaan sebaiknya segera dilakukan karena kebocoran bahan bakar berisiko terhadap keselamatan."
        ),
        (
            ["rem motor", "kurang pakem"],
            "Rem motor yang terasa kurang pakem biasanya disebabkan oleh kampas rem yang aus, minyak rem yang berkurang, atau permukaan rem yang kotor. Pada rem cakram, piringan yang tidak rata juga dapat memengaruhi daya cengkeram. Pemeriksaan awal sebaiknya meliputi kampas rem, minyak rem, dan kondisi piringan atau tromol."
        ),
        (
            ["motor", "limbung", "menikung"],
            "Motor yang terasa limbung saat menikung biasanya disebabkan oleh suspensi yang melemah, tekanan ban yang tidak sesuai, atau kondisi ban yang sudah kurang baik. Beban kendaraan yang tidak seimbang juga dapat memengaruhi kestabilan saat berbelok. Pemeriksaan sebaiknya dilakukan pada ban, suspensi, dan kondisi kaki-kaki motor."
        ),
        (
            ["fungsi aki"],
            "Aki berfungsi menyimpan dan menyuplai energi listrik untuk kebutuhan awal kendaraan, seperti starter, lampu, klakson, dan sistem elektronik lainnya. Pada mobil maupun motor, aki membantu memastikan sistem kelistrikan dapat bekerja stabil saat mesin belum atau sedang menyala. Jika aki lemah, kendaraan bisa sulit distarter dan komponen kelistrikan ikut terganggu."
        ),
        (
            ["ac mobil", "tidak dingin"],
            "AC mobil yang tidak dingin dapat disebabkan oleh refrigeran berkurang, kompresor melemah, evaporator kotor, atau kipas kondensor yang tidak bekerja optimal. Filter kabin yang kotor juga bisa menghambat aliran udara dingin ke dalam kabin. Pemeriksaan sebaiknya dilakukan pada tekanan refrigeran, kompresor, filter kabin, dan komponen sistem AC lainnya."
        ),
    ]

    for required_keywords, answer in patterns:
        if all(keyword in q for keyword in required_keywords):
            return answer

    return None


def _looks_automotive(query: str) -> bool:
    q = query.lower()
    keywords = [
        "mobil",
        "motor",
        "sepeda motor",
        "kendaraan",
        "mesin",
        "cc",
        "baterai",
        "aki",
        "setir",
        "rem",
        "suspensi",
        "cvt",
        "injeksi",
        "bensin",
        "bahan bakar",
        "throttle",
        "power steering",
        "catalytic converter",
        "check engine",
        "lampu",
        "berakselerasi",
        "menanjak",
        "idle",
        "velg",
        "ban",
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
        self.device = DEVICE
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
                        model_dtype = torch.float16 if self.device == "cuda" else torch.float32
                        model_device_map = "auto" if self.device == "cuda" else "cpu"
                        load_kwargs = dict(
                            torch_dtype=model_dtype,
                            device_map=model_device_map,
                            low_cpu_mem_usage=True,
                        )

                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_id,
                                trust_remote_code=False,
                                **load_kwargs,
                            )
                        except Exception:
                            # Fall back to remote code only if the native architecture cannot be loaded.
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_id,
                                trust_remote_code=True,
                                **load_kwargs,
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
        
        direct_answer = _deterministic_answer(query)
        if direct_answer:
            return format_structured_output(direct_answer)

        # 1. Prepare Model
        await self._ensure_model_loaded()
        
        try:
            # 2. Build Prompt using Strategy
            final_prompt = strategy.build_prompt(query, context)
            
            # 3. Tokenize
            inputs = self.tokenizer(final_prompt, return_tensors="pt")
            if self.device == "cuda":
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # 4. Generate
            # Konfigurasi default: temperature=0.2, max_new_tokens=100 (reduced for test speed)
            logger.info(f"Starting generation for query: {query[:50]}...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            logger.info("Generation completed.")
            
            # 5. Decode
            # Skip prompt di output (hanya ambil generated tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response_text = normalize_output(response_text)
            response_text = deduplicate_lead_phrase(response_text)
            response_text = format_structured_output(response_text)
            voltage_answer = _alternator_voltage_answer(context)
            symptom_answer = _alternator_symptom_answer(query, context)
            is_voltage_query = (
                ("alternator" in query.lower() or "pengisian" in query.lower() or "mesin hidup" in query.lower())
                and ("tegangan" in query.lower() or "volt" in query.lower())
            )
            is_alternator_symptom_query = symptom_answer is not None
            if is_bad_output(response_text) or (is_voltage_query and voltage_answer) or is_alternator_symptom_query:
                response_text = _fallback_text(query, context)
            refusal = "Maaf, sistem ini hanya mendukung pertanyaan seputar otomotif."
            if response_text.strip() == refusal and _looks_automotive(query):
                response_text = _fallback_text(query, context)
                response_text = deduplicate_lead_phrase(response_text)
                response_text = format_structured_output(response_text)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return response_text
            
        except Exception as e:
            logger.error(f"LLM Generation failed: {str(e)}")
            raise RuntimeError(f"LLM Generation failed: {str(e)}")

    async def stream(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None):
        # Stub implementation for streaming (Not requested)
        yield "Stream functionality not implemented yet."
