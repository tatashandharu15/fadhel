import csv
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean

import requests
try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    class SmoothingFunction:  # type: ignore[override]
        def __init__(self):
            self.method1 = None


    def sentence_bleu(references, hypothesis, smoothing_function=None):  # type: ignore[override]
        _ = smoothing_function
        ref_tokens = set(references[0]) if references else set()
        hyp_tokens = set(hypothesis or [])
        if not ref_tokens or not hyp_tokens:
            return 0.0
        overlap = len(ref_tokens & hyp_tokens)
        return overlap / max(len(hyp_tokens), 1)


try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]
    util = None  # type: ignore[assignment]

try:
    from rouge_score import rouge_scorer
except ImportError:
    class _SimpleRougeScore:
        def __init__(self, fmeasure: float):
            self.fmeasure = fmeasure


    class _FallbackRougeScorer:
        def __init__(self, metrics: list[str], use_stemmer: bool = True):
            self.metrics = metrics
            self.use_stemmer = use_stemmer

        @staticmethod
        def _tokenize(text: str) -> list[str]:
            return re.findall(r"\w+(?:[.,]\w+)?", text.lower())

        @staticmethod
        def _lcs_length(a: list[str], b: list[str]) -> int:
            if not a or not b:
                return 0
            dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[-1][-1]

        def score(self, target: str, prediction: str) -> dict:
            target_tokens = self._tokenize(target)
            prediction_tokens = self._tokenize(prediction)
            lcs = self._lcs_length(target_tokens, prediction_tokens)
            precision = lcs / len(prediction_tokens) if prediction_tokens else 0.0
            recall = lcs / len(target_tokens) if target_tokens else 0.0
            fmeasure = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            return {"rougeL": _SimpleRougeScore(fmeasure)}


    class _FallbackRougeModule:
        RougeScorer = _FallbackRougeScorer


    rouge_scorer = _FallbackRougeModule()


BASE_DIR = Path(__file__).resolve().parent
API_URL = os.getenv("EVALUATION_API_URL", "http://localhost:8000/v1/chat/completions")
OUTPUT_PATH = BASE_DIR / "results.json"
CSV_OUTPUT_PATH = BASE_DIR / "results.csv"
SIMILARITY_THRESHOLD = 0.35
KEY_FACT_COVERAGE_THRESHOLD = 0.6

STOPWORDS = {
    "a",
    "adalah",
    "agar",
    "akan",
    "atau",
    "bagi",
    "bahwa",
    "banyak",
    "beberapa",
    "by",
    "dan",
    "dari",
    "dengan",
    "di",
    "for",
    "guna",
    "hingga",
    "itu",
    "its",
    "juga",
    "ke",
    "karena",
    "lebih",
    "many",
    "menggunakan",
    "merupakan",
    "of",
    "on",
    "pada",
    "paling",
    "para",
    "secara",
    "seperti",
    "serta",
    "suatu",
    "the",
    "to",
    "umumnya",
    "untuk",
    "yang",
}

EVAL_CASES = [
    {
        "query": "Berapa kapasitas baterai Wuling Air EV?",
        "reference": "Kapasitas baterai Wuling Air EV bervariasi berdasarkan varian, umumnya sekitar 17.3 kWh hingga 26.7 kWh.",
        "expected_keywords": ["wuling", "air", "ev", "17.3", "26.7", "kwh", "baterai"],
    },
    {
        "query": "Apa mesin yang digunakan Honda CR-V?",
        "reference": "Honda CR-V menggunakan mesin bensin 1.5L VTEC Turbo pada banyak varian, dan beberapa varian juga tersedia dalam konfigurasi hybrid e:HEV.",
        "expected_keywords": ["honda", "cr", "v", "1.5l", "vtec", "turbo", "hybrid", "ehev", "mesin"],
    },
    {
        "query": "Bandingkan Honda CR-V dan Toyota Fortuner.",
        "reference": "Honda CR-V cenderung unggul pada kenyamanan kabin dan efisiensi varian hybrid, sedangkan Toyota Fortuner kuat pada karakter SUV ladder frame dan kemampuan medan berat.",
        "expected_keywords": ["honda", "cr", "v", "toyota", "fortuner", "hybrid", "suv", "ladder", "frame"],
    },
    {
        "query": "Apa itu mobil listrik?",
        "reference": "Mobil listrik adalah kendaraan yang digerakkan motor listrik dengan sumber energi utama dari baterai yang dapat diisi ulang.",
        "expected_keywords": ["mobil", "listrik", "kendaraan", "motor", "baterai"],
    },
    {
        "query": "Kenapa mobil sulit dihidupkan saat pagi hari?",
        "reference": "Mobil sulit dihidupkan saat pagi hari biasanya disebabkan oleh aki yang lemah, sistem pengapian yang kurang optimal, atau suplai bahan bakar yang tidak lancar.",
        "expected_keywords": ["mobil", "sulit", "dihidupkan", "pagi", "aki", "pengapian", "bahan", "bakar"],
    },
    {
        "query": "Apa penyebab mobil terasa tersendat saat akselerasi?",
        "reference": "Mobil terasa tersendat saat akselerasi biasanya disebabkan oleh gangguan pada suplai bahan bakar, sistem pengapian, injektor, atau throttle body yang kotor.",
        "expected_keywords": ["mobil", "tersendat", "akselerasi", "bahan", "bakar", "pengapian", "injektor", "throttle"],
    },
    {
        "query": "Kenapa knalpot mobil mengeluarkan asap hitam?",
        "reference": "Asap hitam pada knalpot mobil biasanya menandakan campuran bahan bakar terlalu kaya sehingga pembakaran tidak sempurna.",
        "expected_keywords": ["knalpot", "mobil", "asap", "hitam", "bahan", "bakar", "pembakaran"],
    },
    {
        "query": "Apa penyebab mesin mobil cepat panas saat macet?",
        "reference": "Mesin mobil cepat panas saat macet biasanya disebabkan oleh sistem pendingin yang kurang optimal, seperti radiator kotor, kipas pendingin bermasalah, atau coolant berkurang.",
        "expected_keywords": ["mesin", "mobil", "panas", "macet", "pendingin", "radiator", "kipas", "coolant"],
    },
    {
        "query": "Kenapa setir mobil terasa berat saat diputar?",
        "reference": "Setir mobil terasa berat saat diputar biasanya disebabkan oleh gangguan pada sistem power steering, tekanan ban rendah, atau komponen kemudi yang aus.",
        "expected_keywords": ["setir", "mobil", "berat", "power", "steering", "ban", "kemudi"],
    },
    {
        "query": "Apa fungsi radiator pada mobil?",
        "reference": "Radiator berfungsi membuang panas dari cairan pendingin agar suhu mesin mobil tetap stabil saat bekerja.",
        "expected_keywords": ["radiator", "mobil", "panas", "pendingin", "suhu", "mesin"],
    },
    {
        "query": "Kenapa rem mobil berbunyi saat digunakan?",
        "reference": "Rem mobil berbunyi saat digunakan biasanya disebabkan oleh kampas rem yang menipis, cakram kotor, atau permukaan rem yang tidak rata.",
        "expected_keywords": ["rem", "mobil", "berbunyi", "kampas", "cakram", "kotor"],
    },
    {
        "query": "Apa penyebab aki mobil cepat soak?",
        "reference": "Aki mobil cepat soak biasanya disebabkan oleh sistem pengisian yang tidak normal, usia aki yang menurun, atau adanya arus bocor pada kelistrikan.",
        "expected_keywords": ["aki", "mobil", "soak", "pengisian", "arus", "bocor", "kelistrikan"],
    },
    {
        "query": "Apa fungsi oli mesin pada mobil?",
        "reference": "Oli mesin berfungsi melumasi komponen internal mesin, mengurangi gesekan, membantu pendinginan, dan menjaga kebersihan mesin.",
        "expected_keywords": ["oli", "mesin", "mobil", "melumasi", "gesekan", "pendinginan"],
    },
    {
        "query": "Apa perbedaan transmisi manual dan otomatis pada mobil?",
        "reference": "Transmisi manual membutuhkan perpindahan gigi secara langsung oleh pengemudi, sedangkan transmisi otomatis berpindah gigi secara otomatis sesuai kondisi berkendara.",
        "expected_keywords": ["transmisi", "manual", "otomatis", "gigi", "pengemudi"],
    },
    {
        "query": "Apa fungsi turbo pada mesin mobil?",
        "reference": "Turbo berfungsi meningkatkan jumlah udara yang masuk ke mesin sehingga tenaga dan efisiensi mesin dapat meningkat.",
        "expected_keywords": ["turbo", "mesin", "mobil", "udara", "tenaga", "efisiensi"],
    },
    {
        "query": "Kenapa konsumsi bensin mobil terasa lebih boros dari biasanya?",
        "reference": "Konsumsi bensin mobil yang boros biasanya disebabkan oleh pembakaran yang tidak efisien, filter udara kotor, injektor bermasalah, atau tekanan ban rendah.",
        "expected_keywords": ["bensin", "mobil", "boros", "pembakaran", "filter", "udara", "injektor", "ban"],
    },
    {
        "query": "Apa penyebab mobil bergetar saat idle?",
        "reference": "Mobil bergetar saat idle biasanya disebabkan oleh pembakaran yang tidak stabil, busi kotor, injektor kurang optimal, atau engine mounting melemah.",
        "expected_keywords": ["mobil", "bergetar", "idle", "busi", "injektor", "engine", "mounting"],
    },
    {
        "query": "Apa arti lampu check engine yang menyala?",
        "reference": "Lampu check engine menandakan ada gangguan yang terdeteksi pada sistem mesin, sensor, atau sistem emisi kendaraan.",
        "expected_keywords": ["check", "engine", "gangguan", "mesin", "sensor", "emisi"],
    },
    {
        "query": "Apa perbedaan mobil hybrid dan mobil listrik?",
        "reference": "Mobil hybrid menggabungkan mesin pembakaran dan motor listrik, sedangkan mobil listrik hanya menggunakan motor listrik dengan sumber energi dari baterai.",
        "expected_keywords": ["mobil", "hybrid", "listrik", "mesin", "motor", "baterai"],
    },
    {
        "query": "Apa fungsi catalytic converter pada mobil?",
        "reference": "Catalytic converter berfungsi mengurangi kandungan gas berbahaya pada emisi kendaraan sebelum keluar melalui knalpot.",
        "expected_keywords": ["catalytic", "converter", "mobil", "emisi", "gas", "knalpot"],
    },
    {
        "query": "Kenapa AC mobil tidak dingin?",
        "reference": "AC mobil tidak dingin dapat disebabkan oleh refrigeran berkurang, kompresor melemah, evaporator kotor, atau kipas kondensor bermasalah.",
        "expected_keywords": ["ac", "mobil", "dingin", "refrigeran", "kompresor", "evaporator", "kondensor"],
    },
    {
        "query": "Apa penyebab mobil limbung saat menikung?",
        "reference": "Mobil limbung saat menikung biasanya disebabkan oleh suspensi melemah, tekanan ban tidak sesuai, atau kondisi kaki-kaki yang kurang baik.",
        "expected_keywords": ["mobil", "limbung", "menikung", "suspensi", "ban", "kaki"],
    },
    {
        "query": "Kenapa mobil sulit berakselerasi saat menanjak?",
        "reference": "Mobil sulit berakselerasi saat menanjak biasanya disebabkan oleh tenaga mesin menurun, pembakaran tidak optimal, atau transmisi yang tidak bekerja maksimal.",
        "expected_keywords": ["mobil", "akselerasi", "menanjak", "tenaga", "mesin", "pembakaran", "transmisi"],
    },
    {
        "query": "Apa fungsi power steering pada mobil?",
        "reference": "Power steering berfungsi meringankan putaran setir agar pengemudi lebih mudah mengendalikan mobil.",
        "expected_keywords": ["power", "steering", "mobil", "setir", "mengendalikan"],
    },
    {
        "query": "Kenapa rem motor saya berbunyi berdecit saat digunakan terutama di kecepatan rendah?",
        "reference": "Rem motor yang berbunyi berdecit di kecepatan rendah biasanya disebabkan oleh kampas rem aus, permukaan rem kotor, atau material kampas yang mengeras.",
        "expected_keywords": ["rem", "motor", "berdecit", "kampas", "kotor", "kecepatan", "rendah"],
    },
    {
        "query": "Motor terasa bergetar saat langsam atau idle apa penyebabnya?",
        "reference": "Motor bergetar saat langsam atau idle biasanya disebabkan oleh pembakaran yang tidak stabil, busi lemah, setelan idle tidak tepat, atau dudukan mesin bermasalah.",
        "expected_keywords": ["motor", "bergetar", "langsam", "idle", "busi", "pembakaran"],
    },
    {
        "query": "Kenapa motor sulit dihidupkan saat pagi hari?",
        "reference": "Motor sulit dihidupkan saat pagi hari biasanya disebabkan oleh aki lemah, busi kurang baik, atau suplai bahan bakar yang tidak lancar.",
        "expected_keywords": ["motor", "sulit", "dihidupkan", "pagi", "aki", "busi", "bahan", "bakar"],
    },
    {
        "query": "Motor tersendat saat digas kenapa hal ini bisa terjadi?",
        "reference": "Motor tersendat saat digas biasanya disebabkan oleh karburator atau injektor bermasalah, suplai bahan bakar terganggu, atau busi lemah.",
        "expected_keywords": ["motor", "tersendat", "digas", "karburator", "injektor", "bahan", "bakar", "busi"],
    },
    {
        "query": "Kenapa knalpot motor mengeluarkan asap hitam?",
        "reference": "Asap hitam pada knalpot motor menandakan campuran bahan bakar terlalu kaya sehingga pembakaran tidak sempurna.",
        "expected_keywords": ["knalpot", "motor", "asap", "hitam", "bahan", "bakar", "pembakaran"],
    },
    {
        "query": "Motor terasa kurang bertenaga saat digunakan apa penyebabnya?",
        "reference": "Motor kurang bertenaga biasanya disebabkan oleh filter udara kotor, busi melemah, suplai bahan bakar tidak optimal, atau kompresi mesin menurun.",
        "expected_keywords": ["motor", "kurang", "bertenaga", "filter", "udara", "busi", "bahan", "bakar", "kompresi"],
    },
    {
        "query": "Kenapa lampu motor terlihat redup saat dinyalakan?",
        "reference": "Lampu motor yang redup biasanya menunjukkan masalah pada aki, sistem pengisian, spul, atau regulator.",
        "expected_keywords": ["lampu", "motor", "redup", "aki", "pengisian", "spul", "regulator"],
    },
    {
        "query": "Motor cepat panas saat digunakan dalam perjalanan jauh kenapa?",
        "reference": "Motor cepat panas dalam perjalanan jauh biasanya disebabkan oleh pelumasan yang kurang optimal, kualitas oli menurun, atau sistem pendinginan tidak bekerja maksimal.",
        "expected_keywords": ["motor", "panas", "perjalanan", "jauh", "pelumasan", "oli", "pendinginan"],
    },
    {
        "query": "Kenapa suara mesin motor terdengar lebih kasar dari biasanya?",
        "reference": "Suara mesin motor yang kasar biasanya disebabkan oleh pelumasan kurang baik, oli menurun, atau adanya keausan pada komponen internal mesin.",
        "expected_keywords": ["suara", "mesin", "motor", "kasar", "pelumasan", "oli", "keausan"],
    },
    {
        "query": "Motor terasa berat saat dikendarai apa penyebabnya?",
        "reference": "Motor terasa berat saat dikendarai bisa disebabkan oleh tekanan ban kurang, rem seret, rantai terlalu kencang, atau sistem transmisi kurang optimal.",
        "expected_keywords": ["motor", "berat", "dikendarai", "ban", "rem", "rantai", "transmisi"],
    },
    {
        "query": "Rantai motor berbunyi kasar saat jalan kenapa?",
        "reference": "Rantai motor berbunyi kasar biasanya disebabkan oleh kurang pelumasan, setelan rantai tidak tepat, atau rantai dan gear yang mulai aus.",
        "expected_keywords": ["rantai", "motor", "berbunyi", "kasar", "pelumasan", "gear", "aus"],
    },
    {
        "query": "Motor tidak stabil saat kecepatan tinggi apa penyebabnya?",
        "reference": "Motor tidak stabil saat kecepatan tinggi biasanya disebabkan oleh ban tidak seimbang, velg kurang lurus, atau suspensi yang melemah.",
        "expected_keywords": ["motor", "tidak", "stabil", "kecepatan", "tinggi", "ban", "velg", "suspensi"],
    },
    {
        "query": "Motor brebet saat jalan pelan kenapa bisa begitu?",
        "reference": "Motor brebet saat jalan pelan umumnya disebabkan oleh suplai bahan bakar yang tidak stabil, karburator kotor, atau setelan idle yang kurang tepat.",
        "expected_keywords": ["motor", "brebet", "jalan", "pelan", "bahan", "bakar", "karburator", "idle"],
    },
    {
        "query": "Motor tiba tiba mati saat dikendarai apa penyebabnya?",
        "reference": "Motor yang tiba-tiba mati saat dikendarai bisa disebabkan oleh gangguan pada sistem bahan bakar, pengapian, aki, atau kelistrikan.",
        "expected_keywords": ["motor", "mati", "dikendarai", "bahan", "bakar", "pengapian", "aki", "kelistrikan"],
    },
    {
        "query": "Starter elektrik motor tidak berfungsi kenapa?",
        "reference": "Starter elektrik motor yang tidak berfungsi biasanya disebabkan oleh aki lemah, relay starter bermasalah, atau dinamo starter yang tidak bekerja optimal.",
        "expected_keywords": ["starter", "elektrik", "motor", "aki", "relay", "dinamo"],
    },
    {
        "query": "Kenapa motor sulit dinyalakan saat kondisi panas?",
        "reference": "Motor sulit dinyalakan saat kondisi panas biasanya disebabkan oleh sistem pengapian atau bahan bakar yang tidak bekerja optimal saat suhu mesin tinggi.",
        "expected_keywords": ["motor", "sulit", "dinyalakan", "panas", "pengapian", "bahan", "bakar"],
    },
    {
        "query": "Konsumsi bensin motor terasa lebih boros dari biasanya kenapa?",
        "reference": "Konsumsi bensin motor yang boros biasanya disebabkan oleh pembakaran tidak efisien, filter udara kotor, injektor bermasalah, atau gaya berkendara yang agresif.",
        "expected_keywords": ["bensin", "motor", "boros", "pembakaran", "filter", "udara", "injektor"],
    },
    {
        "query": "Tercium bau bensin dari motor apa penyebabnya?",
        "reference": "Bau bensin dari motor biasanya menandakan adanya kebocoran pada sistem bahan bakar, selang bensin, atau campuran bahan bakar yang terlalu kaya.",
        "expected_keywords": ["bau", "bensin", "motor", "kebocoran", "selang", "bahan", "bakar"],
    },
    {
        "query": "Rem motor terasa kurang pakem saat digunakan kenapa?",
        "reference": "Rem motor yang kurang pakem biasanya disebabkan oleh kampas rem aus, minyak rem berkurang, atau permukaan rem yang kotor.",
        "expected_keywords": ["rem", "motor", "kurang", "pakem", "kampas", "minyak", "kotor"],
    },
    {
        "query": "Motor terasa limbung saat menikung apa penyebabnya?",
        "reference": "Motor terasa limbung saat menikung biasanya disebabkan oleh suspensi yang melemah, tekanan ban tidak sesuai, atau kondisi ban yang sudah kurang baik.",
        "expected_keywords": ["motor", "limbung", "menikung", "suspensi", "ban"],
    },
    {
        "query": "Apa perbedaan rem cakram dan rem tromol pada kendaraan?",
        "reference": "Rem cakram menggunakan piringan dan kaliper sehingga respons pengereman biasanya lebih baik, sedangkan rem tromol memakai mekanisme tertutup yang umumnya lebih sederhana dan ekonomis.",
        "expected_keywords": ["rem", "cakram", "tromol", "piringan", "kaliper", "pengereman"],
    },
    {
        "query": "Apa fungsi aki pada mobil dan motor?",
        "reference": "Aki berfungsi menyimpan dan menyuplai energi listrik untuk starter, lampu, klakson, dan sistem kelistrikan kendaraan.",
        "expected_keywords": ["aki", "mobil", "motor", "energi", "listrik", "starter", "lampu", "kelistrikan"],
    },
    {
        "query": "Kenapa kendaraan terasa bergetar saat kecepatan tinggi?",
        "reference": "Kendaraan yang bergetar saat kecepatan tinggi biasanya disebabkan oleh ban yang tidak seimbang, velg kurang lurus, atau suspensi dan bearing yang bermasalah.",
        "expected_keywords": ["kendaraan", "bergetar", "kecepatan", "tinggi", "ban", "velg", "suspensi", "bearing"],
    },
    {
        "query": "Apa penyebab suara mesin menjadi kasar setelah dipakai lama?",
        "reference": "Suara mesin yang menjadi kasar setelah dipakai lama biasanya disebabkan oleh kualitas oli menurun, pelumasan kurang optimal, atau keausan komponen mesin.",
        "expected_keywords": ["suara", "mesin", "kasar", "oli", "pelumasan", "keausan"],
    },
    {
        "query": "Bagaimana cara kerja sistem injeksi bahan bakar?",
        "reference": "Sistem injeksi bahan bakar bekerja dengan mengatur jumlah dan waktu penyemprotan bahan bakar secara presisi berdasarkan data sensor dan kendali ECU.",
        "expected_keywords": ["injeksi", "bahan", "bakar", "penyemprotan", "sensor", "ecu"],
    },
    {
        "query": "Apa fungsi suspensi pada mobil dan motor?",
        "reference": "Suspensi berfungsi meredam guncangan dari permukaan jalan agar kendaraan tetap nyaman dan stabil saat digunakan.",
        "expected_keywords": ["suspensi", "mobil", "motor", "guncangan", "nyaman", "stabil"],
    },
    {
        "query": "Kenapa kendaraan terasa kurang responsif saat gas ditarik?",
        "reference": "Kendaraan yang kurang responsif saat gas ditarik biasanya disebabkan oleh suplai bahan bakar yang kurang lancar, throttle body kotor, atau sistem pengapian yang tidak optimal.",
        "expected_keywords": ["kendaraan", "kurang", "responsif", "gas", "bahan", "bakar", "throttle", "pengapian"],
    },
    {
        "query": "Apa penyebab kendaraan cepat kehabisan bahan bakar?",
        "reference": "Kendaraan cepat kehabisan bahan bakar biasanya disebabkan oleh pembakaran yang tidak efisien, gaya berkendara agresif, atau masalah pada sistem bahan bakar dan tekanan ban.",
        "expected_keywords": ["kendaraan", "kehabisan", "bahan", "bakar", "pembakaran", "ban"],
    },
    {
        "query": "Apa fungsi filter udara pada mesin kendaraan?",
        "reference": "Filter udara berfungsi menyaring kotoran dari udara sebelum masuk ke mesin agar proses pembakaran tetap bersih dan efisien.",
        "expected_keywords": ["filter", "udara", "mesin", "kendaraan", "kotoran", "pembakaran"],
    },
    {
        "query": "Kenapa kendaraan sulit dinyalakan setelah terkena hujan?",
        "reference": "Kendaraan yang sulit dinyalakan setelah terkena hujan biasanya disebabkan oleh kelembapan pada sistem pengapian atau gangguan pada komponen kelistrikan.",
        "expected_keywords": ["kendaraan", "sulit", "dinyalakan", "hujan", "kelembapan", "pengapian", "kelistrikan"],
    },
]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _extract_answer_text(data: dict) -> str:
    answer = data.get("answer")
    if isinstance(answer, dict):
        id_text = str(answer.get("id", "")).strip()
        if id_text:
            return id_text
    return str(data.get("response", "")).strip()


def _extract_latency_ms(data: dict) -> float:
    trace = data.get("trace", {}) or {}
    if isinstance(trace, dict):
        trace_latency = trace.get("latency_ms")
        if trace_latency is not None:
            return float(trace_latency or 0.0)
    return float(data.get("latency_ms", 0.0) or 0.0)


@lru_cache(maxsize=1)
def _get_semantic_similarity_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower())


def _normalize_keyword(token: str) -> str:
    return token.replace(",", ".").replace(":", "")


def _extract_keywords(text: str) -> set[str]:
    keywords = set()
    for token in _tokenize(text):
        normalized = _normalize_keyword(token)
        if normalized in STOPWORDS:
            continue
        if len(normalized) <= 2 and not any(ch.isdigit() for ch in normalized):
            continue
        keywords.add(normalized)
    return keywords


def _get_expected_keywords(case: dict) -> set[str]:
    manual_keywords = {_normalize_keyword(k.lower()) for k in case.get("expected_keywords", [])}
    if manual_keywords:
        return manual_keywords
    return _extract_keywords(case["reference"])


def _keyword_confusion(expected_keywords: set[str], generated_keywords: set[str]) -> tuple[int, int, int]:
    tp = len(expected_keywords & generated_keywords)
    fp = len(generated_keywords - expected_keywords)
    fn = len(expected_keywords - generated_keywords)
    return tp, fp, fn


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _contains_expected_key_facts(expected_keywords: set[str], generated_keywords: set[str]) -> bool:
    if not expected_keywords:
        return False
    coverage = _safe_ratio(len(expected_keywords & generated_keywords), len(expected_keywords))
    return coverage >= KEY_FACT_COVERAGE_THRESHOLD


def _fallback_semantic_similarity(reference: str, answer: str) -> float:
    reference_keywords = _extract_keywords(reference)
    answer_keywords = _extract_keywords(answer)
    if not reference_keywords or not answer_keywords:
        return 0.0
    overlap = len(reference_keywords & answer_keywords)
    union = len(reference_keywords | answer_keywords)
    return _safe_ratio(overlap, union)


def _calculate_semantic_similarity(reference: str, answer: str) -> float:
    if not reference.strip() or not answer.strip():
        return 0.0

    model = _get_semantic_similarity_model()
    if model is None or util is None:
        return _clamp(_fallback_semantic_similarity(reference, answer))

    try:
        embeddings = model.encode([reference, answer], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return _clamp(similarity)
    except Exception:
        return _clamp(_fallback_semantic_similarity(reference, answer))


def _join_source_contents(sources: list[dict]) -> str:
    parts = []
    for source in sources or []:
        if isinstance(source, dict):
            content = str(source.get("content", "")).strip()
            if content:
                parts.append(content)
    return "\n".join(parts)


def _calculate_answer_relevancy(query: str, answer: str) -> float:
    return _calculate_semantic_similarity(query, answer)


def _calculate_faithfulness(answer: str, sources: list[dict]) -> float:
    source_text = _join_source_contents(sources)
    if not answer.strip() or not source_text.strip():
        return 0.0

    source_similarity = _calculate_semantic_similarity(source_text, answer)
    answer_keywords = _extract_keywords(answer)
    source_keywords = _extract_keywords(source_text)
    support_ratio = _safe_ratio(len(answer_keywords & source_keywords), len(answer_keywords)) if answer_keywords else 0.0
    return _clamp((0.6 * source_similarity) + (0.4 * support_ratio))


def _calculate_answer_correctness(semantic_similarity: float, precision: float, recall: float, f1_score: float) -> float:
    correctness = (
        (0.5 * semantic_similarity)
        + (0.2 * precision)
        + (0.1 * recall)
        + (0.2 * f1_score)
    )
    return _clamp(correctness)


def _calculate_semantic_metrics(case: dict, answer: str, rouge_l: float, semantic_similarity: float) -> dict:
    expected_keywords = _get_expected_keywords(case)
    generated_keywords = _extract_keywords(answer)

    tp, fp, fn = _keyword_confusion(expected_keywords, generated_keywords)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_score = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    similarity = max(semantic_similarity, rouge_l, recall)
    is_correct = similarity >= SIMILARITY_THRESHOLD or _contains_expected_key_facts(expected_keywords, generated_keywords)
    accuracy = 1.0 if is_correct else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "expected_keywords": sorted(expected_keywords),
        "generated_keywords": sorted(generated_keywords),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _calculate_confidence_score(retrieval_score: float, rouge_l: float, bleu: float) -> float:
    confidence = (0.5 * retrieval_score) + (0.25 * rouge_l) + (0.25 * bleu)
    return _clamp(confidence)


def _write_csv(rows: list[dict]) -> None:
    csv_path = _resolve_writable_output_path(CSV_OUTPUT_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "response",
        "bleu",
        "rougeL",
        "semantic_similarity",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confidence_score",
        "latency_ms",
        "retrieval_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _resolve_writable_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def _order_detail_row(row: dict) -> dict:
    return {
        "query": row.get("query", ""),
        "reference": row.get("reference", ""),
        "response": row.get("response", ""),
        "bleu": row.get("bleu", 0.0),
        "rougeL": row.get("rougeL", 0.0),
        "semantic_similarity": row.get("semantic_similarity", 0.0),
        "faithfulness": row.get("faithfulness", 0.0),
        "answer_relevancy": row.get("answer_relevancy", 0.0),
        "answer_correctness": row.get("answer_correctness", 0.0),
        "accuracy": row.get("accuracy", 0.0),
        "precision": row.get("precision", 0.0),
        "recall": row.get("recall", 0.0),
        "f1_score": row.get("f1_score", 0.0),
        "confidence_score": row.get("confidence_score", 0.0),
        "latency_ms": row.get("latency_ms", 0.0),
        "retrieval_score": row.get("retrieval_score", 0.0),
        "expected_keywords": row.get("expected_keywords", []),
        "generated_keywords": row.get("generated_keywords", []),
        "tp": row.get("tp", 0),
        "fp": row.get("fp", 0),
        "fn": row.get("fn", 0),
        "error": row.get("error"),
    }


def _order_summary(summary: dict) -> dict:
    return {
        "avg_bleu": summary.get("avg_bleu", 0.0),
        "avg_rougeL": summary.get("avg_rougeL", 0.0),
        "avg_semantic_similarity": summary.get("avg_semantic_similarity", 0.0),
        "avg_faithfulness": summary.get("avg_faithfulness", 0.0),
        "avg_answer_relevancy": summary.get("avg_answer_relevancy", 0.0),
        "avg_answer_correctness": summary.get("avg_answer_correctness", 0.0),
        "avg_accuracy": summary.get("avg_accuracy", 0.0),
        "avg_precision": summary.get("avg_precision", 0.0),
        "avg_recall": summary.get("avg_recall", 0.0),
        "avg_f1_score": summary.get("avg_f1_score", 0.0),
        "avg_confidence_score": summary.get("avg_confidence_score", 0.0),
        "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
        "avg_retrieval_score": summary.get("avg_retrieval_score", 0.0),
    }


def evaluate() -> dict:
    smoothie = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []

    for case in EVAL_CASES:
        payload = {
            "query": case["query"],
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "use_rag": True,
        }
        row = {
            "query": case["query"],
            "reference": case["reference"],
            "response": "",
            "bleu": 0.0,
            "rougeL": 0.0,
            "semantic_similarity": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confidence_score": 0.0,
            "latency_ms": 0.0,
            "retrieval_score": 0.0,
            "error": None,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            answer = _extract_answer_text(data)
            sources = data.get("sources", []) or []
            latency_ms = _extract_latency_ms(data)
            retrieval_score = 0.0
            if sources and isinstance(sources[0], dict):
                retrieval_score = float(sources[0].get("score", 0.0) or 0.0)

            ref_tokens = case["reference"].split()
            ans_tokens = answer.split() if answer else []
            bleu = sentence_bleu([ref_tokens], ans_tokens, smoothing_function=smoothie) if ans_tokens else 0.0
            rouge_l = rouge.score(case["reference"], answer)["rougeL"].fmeasure if answer else 0.0
            semantic_similarity = _calculate_semantic_similarity(case["reference"], answer)
            semantic_metrics = _calculate_semantic_metrics(case, answer, rouge_l, semantic_similarity)
            faithfulness = _calculate_faithfulness(answer, sources)
            answer_relevancy = _calculate_answer_relevancy(case["query"], answer)
            answer_correctness = _calculate_answer_correctness(
                semantic_similarity,
                semantic_metrics["precision"],
                semantic_metrics["recall"],
                semantic_metrics["f1_score"],
            )
            confidence_score = _calculate_confidence_score(retrieval_score, rouge_l, bleu)

            row["response"] = answer
            row["bleu"] = float(bleu)
            row["rougeL"] = float(rouge_l)
            row["semantic_similarity"] = float(semantic_similarity)
            row["faithfulness"] = float(faithfulness)
            row["answer_relevancy"] = float(answer_relevancy)
            row["answer_correctness"] = float(answer_correctness)
            row["accuracy"] = semantic_metrics["accuracy"]
            row["precision"] = semantic_metrics["precision"]
            row["recall"] = semantic_metrics["recall"]
            row["f1_score"] = semantic_metrics["f1_score"]
            row["confidence_score"] = confidence_score
            row["latency_ms"] = latency_ms
            row["retrieval_score"] = retrieval_score
            row["expected_keywords"] = semantic_metrics["expected_keywords"]
            row["generated_keywords"] = semantic_metrics["generated_keywords"]
            row["tp"] = semantic_metrics["tp"]
            row["fp"] = semantic_metrics["fp"]
            row["fn"] = semantic_metrics["fn"]
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    success = [r for r in rows if r["error"] is None]
    summary = {
        "avg_bleu": float(mean([r["bleu"] for r in success])) if success else 0.0,
        "avg_rougeL": float(mean([r["rougeL"] for r in success])) if success else 0.0,
        "avg_semantic_similarity": float(mean([r["semantic_similarity"] for r in success])) if success else 0.0,
        "avg_faithfulness": float(mean([r["faithfulness"] for r in success])) if success else 0.0,
        "avg_answer_relevancy": float(mean([r["answer_relevancy"] for r in success])) if success else 0.0,
        "avg_answer_correctness": float(mean([r["answer_correctness"] for r in success])) if success else 0.0,
        "avg_accuracy": float(mean([r["accuracy"] for r in success])) if success else 0.0,
        "avg_precision": float(mean([r["precision"] for r in success])) if success else 0.0,
        "avg_recall": float(mean([r["recall"] for r in success])) if success else 0.0,
        "avg_f1_score": float(mean([r["f1_score"] for r in success])) if success else 0.0,
        "avg_confidence_score": float(mean([r["confidence_score"] for r in success])) if success else 0.0,
        "avg_latency_ms": float(mean([r["latency_ms"] for r in success])) if success else 0.0,
        "avg_retrieval_score": float(mean([r["retrieval_score"] for r in success])) if success else 0.0,
    }
    json_output_path = _resolve_writable_output_path(OUTPUT_PATH)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = [_order_detail_row(row) for row in rows]
    ordered_summary = _order_summary(summary)
    ordered_result = {
        **ordered_summary,
        "total_queries": len(rows),
        "successful_queries": len(success),
        "failed_queries": len(rows) - len(success),
        "summary": ordered_summary,
        "details": ordered_rows,
    }
    ordered_result["output_files"] = {
        "json": str(json_output_path),
        "csv": str(_resolve_writable_output_path(CSV_OUTPUT_PATH)),
    }
    json_output_path.write_text(json.dumps(ordered_result, indent=2, ensure_ascii=False))
    _write_csv(ordered_rows)
    return ordered_result


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
