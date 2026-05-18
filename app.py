# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO v3.2 — RAG + Agentes + Retroalimentación Vectorial      ║
# ║  EISC — Universidad del Valle, Cali, Colombia                             ║
# ║  CORRECCIONES: DeepSeek device_map robusto, error handling mejorado       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import streamlit as st
import os
import time
import json
import hashlib
import unicodedata
import base64
import uuid
import re
import tempfile
import asyncio
import concurrent.futures
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Generator, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from rank_bm25 import BM25Okapi
import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# DIAGNÓSTICO GLOBAL — Capturar errores de import
# ─────────────────────────────────────────────
_IMPORT_ERRORS = []
try:
    import torch
except Exception as e:
    _IMPORT_ERRORS.append(f"torch: {e}")
    torch = None

# ─────────────────────────────────────────────
# M11 · SEGURIDAD
# ─────────────────────────────────────────────
USERS_HASHED = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "raul":  hashlib.sha256("eisc2025".encode()).hexdigest(),
}
_request_counts: dict[str, list] = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 15
MAX_QUERY_LENGTH = 2000
INJECTION_PATTERNS = [
    r"ignore (?:previous|all) instructions",
    r"forget (?:your|the) (?:system )?prompt",
    r"act as (?:if )?you (?:are|were)",
    r"you are now",
    r"disregard (?:your|all) (?:previous )?",
    r"jailbreak",
]

def verify_password(user: str, password: str) -> bool:
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS_HASHED.get(user) == hashed

def check_rate_limit(user: str) -> bool:
    now = datetime.now().timestamp()
    _request_counts[user] = [t for t in _request_counts[user] if now - t < 60]
    if len(_request_counts[user]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    _request_counts[user].append(now)
    return True

def sanitize_query(query: str) -> str:
    for pattern in INJECTION_PATTERNS:
        query = re.sub(pattern, "[eliminado]", query, flags=re.IGNORECASE)
    return query[:MAX_QUERY_LENGTH].strip()

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
def login():
    st.sidebar.title("🔐 Acceso a ChatAcredita (EISC)")
    user = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if verify_password(user, password):
            st.session_state.auth = True
            st.session_state.user = user
            st.rerun()
        else:
            st.sidebar.error("❌ Credenciales incorrectas")

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    login()
    st.stop()

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChatAcredita PRO v3.2 - EISC-Univalle",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
COLLECTION_NAME        = "acreditacion"
FEEDBACK_COLLECTION    = "feedback_acreditacion"
EVAL_COLLECTION        = "evaluaciones_chatacredita"
TOP_K                  = 15
TOP_K_FINAL            = 10
HALLUCINATION_THRESHOLD = 0.4

# ─────────────────────────────────────────────
# UTILIDADES GENERALES
# ─────────────────────────────────────────────
def get_base64_image(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.lower().split())

def clean_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {}

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# ─────────────────────────────────────────────
# CSS + HEADER
# ─────────────────────────────────────────────
st.markdown("""
<style>
header {visibility: hidden;}
.custom-header {
    position: fixed; top: 0; left: 0; right: 0; height: 64px;
    background: linear-gradient(90deg, #DC143C, #8B0000);
    display: flex; align-items: center; justify-content: center;
    z-index: 9999; color: white; font-weight: 600; font-size: 1.05em;
}
.main { padding-top: 74px; }
.footer {
    position: fixed; bottom: 60px; left: 0; right: 0;
    text-align: center; font-size: 11px; color: #999;
}
.thinking-avatar {
    position: fixed; bottom: 85px; right: 18px;
    background: white; padding: 7px 12px; border-radius: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
    display: flex; align-items: center; gap: 8px; z-index: 9999;
    font-size: 0.9em;
}
.avatar-img { border-radius: 50%; width: 26px; height: 26px; }
.status-analizando  { background:#e3f2fd; border-left:3px solid #2196f3; color:#1565c0; }
.status-expandiendo { background:#ede7f6; border-left:3px solid #7e57c2;  color:#4527a0; }
.status-recuperando { background:#e8f5e9; border-left:3px solid #4caf50; color:#2e7d32; }
.status-reranking   { background:#fff8e1; border-left:3px solid #ffc107;  color:#f57f17; }
.status-generando   { background:#f3e5f5; border-left:3px solid #9c27b0; color:#4a148c; }
.status-evaluando   { background:#fce4ec; border-left:3px solid #e91e63;  color:#880e4f; }
.status-corrigiendo {
    background:#fff3e0; border-left:4px solid #ff9800; color:#e65100;
    animation: pulse 1.5s infinite;
}
.status-listo {
    background:#e8f5e9; border-left:4px solid #4caf50; color:#2e7d32;
    box-shadow: 0 0 12px rgba(76,175,80,0.4);
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0   rgba(255,152,0,0.4); }
    70%  { box-shadow: 0 0 0  8px rgba(255,152,0,0);   }
    100% { box-shadow: 0 0 0 0   rgba(255,152,0,0);   }
}
.source-badge {
    display: inline-block; background:#e3f2fd; color:#1976d2;
    padding: 3px 8px; border-radius: 12px; font-size: 0.84em;
    margin: 2px; border: 1px solid #bbdefb;
}
.sources-container {
    margin-top: 14px; padding: 11px;
    background: #f8fdff; border-left: 3px solid #2196f3;
    border-radius: 0 8px 8px 0;
}
.confidence-bar {
    height: 6px; border-radius: 3px; margin-top: 6px;
    background: linear-gradient(90deg, #4caf50, #ffeb3b, #f44336);
}
.quality-badge  {
    display: inline-block; font-size: 0.78em;
    padding: 2px 7px; border-radius: 10px; margin-left: 6px;
}
.q-high  { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.q-med   { background:#fff8e1; color:#f57f17; border:1px solid #ffe082; }
.q-low   { background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; }
.feedback-indicator {
    display:inline-block; background:#fff3e0; color:#e65100;
    padding:2px 8px; border-radius:10px; font-size:0.8em;
    margin-left:8px; border:1px solid #ffcc80;
}
.rating-btn { cursor: pointer; font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
🎓 ChatAcredita PRO v3.2 — EISC (Universidad del Valle)
&nbsp;·&nbsp; RAG + Agentes + Qwen 7B local
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONEXIÓN APIs — Groq
# ─────────────────────────────────────────────
OPENAI_API_KEY  = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL   = "llama-3.3-70b-versatile"
FAST_MODEL      = "llama-3.3-70b-versatile"
groq_available = False
try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    _ = client.models.list()
    groq_available = True
    st.sidebar.success(f"✅ Groq: {DEFAULT_MODEL}")
except Exception as e:
    st.sidebar.error(f"❌ Groq: {str(e)[:80]}")
    st.sidebar.info("Verifica OPENAI_API_KEY en Secrets")

# ─────────────────────────────────────────────
# CONEXIÓN Qdrant
# ─────────────────────────────────────────────
try:
    qdrant = QdrantClient(
        url=get_secret("QDRANT_URL", "").strip(),
        api_key=get_secret("QDRANT_API_KEY", "").strip(),
    )
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada")
        st.stop()

    for col in [FEEDBACK_COLLECTION, EVAL_COLLECTION]:
        if col not in existing:
            qdrant.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
    st.sidebar.success(f"✅ Qdrant: {COLLECTION_NAME}")
except Exception as e:
    st.sidebar.error(f"❌ Qdrant: {str(e)[:80]}")
    st.stop()

# ─────────────────────────────────────────────
# MODELOS DE EMBEDDINGS + RERANKER
# ─────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3", device="cpu")
embedder = load_embedder()
st.sidebar.success("✅ Embeddings + Reranker: BGE-M3 (1024d)")

# ─────────────────────────────────────────────
# M13 · BACKEND LOCAL: Qwen 2.5-7B
# ─────────────────────────────────────────────
QWEN_MODEL_ID     = "raulgdp/qwen2.5-7b-acredita-cna-col"
DEEPSEEK_MODEL_ID = "raulgdp/deepseek14b-acredita"

@st.cache_resource(show_spinner="🤖 Cargando Qwen 7B acreditación CNA...")
def load_qwen_local():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem >= 14:
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_ID,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError:
                model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

def generate_qwen_response(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 1000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> Generator[str, None, None]:
    import torch
    try:
        qwen_model, qwen_tokenizer = load_qwen_local()
        if qwen_model is None:
            yield "❌ Error cargando modelo Qwen"
            return

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            output = qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.01,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=qwen_tokenizer.eos_token_id,
            )

        response = qwen_tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        for word in response.split(" "):
            yield word + " "
    except Exception as e:
        print(f"🔥 ERROR Qwen: {traceback.format_exc()}")
        yield f"⚠️ Error Qwen local: {str(e)[:200]}"

def generate_qwen_full(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 900,
) -> str:
    return "".join(generate_qwen_response(system_msg, user_msg, max_new_tokens))

# ─────────────────────────────────────────────
# DEEPSEEK 14B — VERSIÓN ROBUSTA CON MANEJO DE ERRORES
# ─────────────────────────────────────────────
_deepseek_load_error = None

@st.cache_resource(show_spinner="🤖 Cargando DeepSeek 14B acreditación CNA...")
def load_deepseek_local():
    global _deepseek_load_error
    _deepseek_load_error = None
    
    # 🛡️ Verificar que torch está disponible
    if torch is None:
        _deepseek_load_error = "PyTorch no está instalado o falló la importación"
        raise RuntimeError(_deepseek_load_error)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except Exception as e:
        _deepseek_load_error = f"Error cargando tokenizer: {str(e)}"
        raise RuntimeError(_deepseek_load_error)

    try:
        if torch.cuda.is_available():
            # GPU: usar 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU: SIN device_map, SIN quantization
            model = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        _deepseek_load_error = f"Error cargando modelo: {str(e)}\n{traceback.format_exc()}"
        raise RuntimeError(_deepseek_load_error)


def generate_deepseek_response(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 1000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> Generator[str, None, None]:
    try:
        ds_model, ds_tokenizer = load_deepseek_local()
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        prompt = ds_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = ds_tokenizer(prompt, return_tensors="pt")
        
        # ✅ Mover inputs al dispositivo correcto
        if torch.cuda.is_available():
            if hasattr(ds_model, 'hf_device_map'):
                first_device = next(iter(ds_model.hf_device_map.values()))
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            output = ds_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.01,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=ds_tokenizer.eos_token_id,
            )

        response = ds_tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        for word in response.split(" "):
            yield word + " "

    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"🔥 ERROR DeepSeek: {error_detail}")
        yield f"⚠️ Error DeepSeek 14B: {str(e)[:200]}"


def generate_deepseek_full(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 900,
) -> str:
    return "".join(generate_deepseek_response(system_msg, user_msg, max_new_tokens))


def _is_deepseek() -> bool:
    return st.session_state.get("use_deepseek", False)

def _is_qwen() -> bool:
    return st.session_state.get("use_qwen", False) and not st.session_state.get("use_deepseek", False)

def _local_generate_response(system_msg, user_msg, max_new_tokens=1000) -> Generator[str, None, None]:
    """Despacha al backend local seleccionado (Qwen o DeepSeek)."""
    if _is_deepseek():
        yield from generate_deepseek_response(system_msg, user_msg, max_new_tokens)
    else:
        yield from generate_qwen_response(system_msg, user_msg, max_new_tokens)

def _local_generate_full(system_msg, user_msg, max_new_tokens=900) -> str:
    """Genera respuesta completa con el backend local seleccionado."""
    if _is_deepseek():
        return generate_deepseek_full(system_msg, user_msg, max_new_tokens)
    return generate_qwen_full(system_msg, user_msg, max_new_tokens)

# ─────────────────────────────────────────────
# M4 · MEMORIA CONVERSACIONAL
# ─────────────────────────────────────────────
class ConversationMemory:
    def __init__(self, max_turns: int = 6, max_summary_tokens: int = 150):
        self.max_turns = max_turns
        self.max_summary_tokens = max_summary_tokens

    def get_context(self, messages: list) -> str:
        clean = [m for m in messages if m.get("role") in ("user", "assistant")]
        recent = clean[-(self.max_turns * 2):]
        if len(clean) <= self.max_turns * 2:
            return self._format(recent)
        old = clean[:-(self.max_turns * 2)]
        summary = self._summarize(old)
        return f"[Resumen diálogo anterior]: {summary}\n\n" + self._format(recent)

    def _summarize(self, messages: list) -> str:
        dialog = self._format(messages)
        try:
            r = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": f"Resume en 2 oraciones este diálogo sobre acreditación EISC:\n{dialog}"}],
                temperature=0, max_tokens=self.max_summary_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return "[historial previo disponible]"

    def _format(self, messages: list) -> str:
        lines = []
        for m in messages:
            role = "Usuario" if m["role"] == "user" else "Asistente"
            content = re.sub(r"<[^>]+>", " ", m.get("content", ""))[:400]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

memory = ConversationMemory()

# ─────────────────────────────────────────────
# M1 · QUERY REWRITING + HyDE
# ─────────────────────────────────────────────
def rewrite_query(query: str, memory_ctx: str) -> dict:
    prompt_text = f"""Eres un experto en acreditación universitaria colombiana (CNA).
Contexto conversacional reciente:
{memory_ctx[-600:] if memory_ctx else 'Sin historial previo.'}
Query del usuario: {query}
Genera un JSON con:
"rewritten": reformulación más precisa y técnica de la query (string)
"hyde": párrafo de 2-3 oraciones que podría aparecer en un documento de acreditación respondiendo esta query (string)
"keywords": lista de 4-6 términos clave para búsqueda BM25 (list)
"lang": idioma detectado "es" o "en" (string)
Solo JSON sin markdown."""
    
    if _is_qwen() or _is_deepseek():
        try:
            raw = _local_generate_full("Eres un experto en acreditación CNA. Responde solo con JSON.", prompt_text, max_new_tokens=350)
            data = clean_json(raw)
            return {"rewritten": data.get("rewritten", query), "hyde": data.get("hyde", ""), "keywords": data.get("keywords", []), "lang": data.get("lang", "es")}
        except Exception:
            return {"rewritten": query, "hyde": "", "keywords": [], "lang": "es"}

    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt_text}], temperature=0, max_tokens=350)
        data = clean_json(r.choices[0].message.content)
        return {"rewritten": data.get("rewritten", query), "hyde": data.get("hyde", ""), "keywords": data.get("keywords", []), "lang": data.get("lang", "es")}
    except Exception:
        return {"rewritten": query, "hyde": "", "keywords": [], "lang": "es"}

# ─────────────────────────────────────────────
# M2 · ÍNDICE BM25 REAL
# ─────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def build_bm25_index() -> tuple:
    all_texts, all_ids, all_sources = [], [], []
    offset = None
    try:
        while True:
            result = qdrant.scroll(collection_name=COLLECTION_NAME, limit=200, offset=offset, with_payload=True, with_vectors=False)
            for point in result[0]:
                if point.payload and point.payload.get("text"):
                    all_texts.append(point.payload["text"])
                    all_ids.append(point.id)
                    all_sources.append(point.payload.get("source", "desconocido"))
            offset = result[1]
            if offset is None:
                break
        tokenized = [normalize_text(t).split() for t in all_texts]
        bm25 = BM25Okapi(tokenized)
        return bm25, all_texts, all_ids, all_sources
    except Exception:
        return None, [], [], []

# ─────────────────────────────────────────────
# M2 + M3 · BÚSQUEDA HÍBRIDA CON RRF + RERANKER
# ─────────────────────────────────────────────
def hybrid_search_rrf(query: str, query_variants: list[str], use_feedback: bool = False, k_rrf: int = 60) -> list[dict]:
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    rrf_scores: dict[str, float] = {}
    id_to_payload: dict[str, dict] = {}
    
    for q in query_variants:
        try:
            emb = embedder.encode([q], normalize_embeddings=True)[0]
            results = qdrant.query_points(collection_name=collection, query=emb.tolist(), limit=TOP_K, with_payload=True).points
            for rank, r in enumerate(results):
                pid = str(r.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
                if r.payload:
                    id_to_payload[pid] = r.payload
        except Exception:
            pass

    if not use_feedback:
        bm25, bm25_texts, bm25_ids, bm25_sources = build_bm25_index()
        if bm25 is not None:
            tokens = normalize_text(query).split()
            scores = bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]
            for rank, (idx, score) in enumerate(ranked):
                if score > 0 and idx < len(bm25_ids):
                    pid = str(bm25_ids[idx])
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
                    if pid not in id_to_payload:
                        id_to_payload[pid] = {"text": bm25_texts[idx], "source": bm25_sources[idx]}

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    results = []
    for pid, score in sorted_ids:
        payload = id_to_payload.get(pid, {})
        if payload.get("text"):
            results.append({"id": pid, "text": payload["text"], "source": payload.get("source", "desconocido"), "score": round(score, 4)})
    return results

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    for r in results:
        base_score = r.get("rrf_score", r.get("score", 0.0))
        if "| " in r["text"] and r["text"].count("| ") > 6:
            base_score += 0.003
        if "feedback" in r.get("source", "").lower():
            base_score += 0.005
        q_words = set(normalize_text(query).split())
        chunk_words = set(r["text"].lower().split())
        overlap = len(q_words & chunk_words)
        if overlap > 3:
            base_score += 0.001 * overlap
        r["rerank_score"] = base_score
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:TOP_K_FINAL]

# ─────────────────────────────────────────────
# M5 · ROUTER DE AGENTES
# ─────────────────────────────────────────────
AGENT_TYPES = {
    "estadistica": "Pregunta sobre números, tasas, porcentajes, cantidades, rankings",
    "normativa": "Pregunta sobre reglamentos, resoluciones, leyes, artículos, normas CNA",
    "proceso": "Pregunta sobre pasos, procedimientos, flujos, cómo hacer algo",
    "comparacion": "Pregunta que compara dos o más elementos, criterios o periodos",
    "sintesis": "Pregunta de resumen, conclusión o visión general de múltiples aspectos",
    "general": "Cualquier otra pregunta sobre acreditación EISC",
}

AGENT_PROMPTS = {
    "estadistica": "Presenta los datos numéricos con precisión. Usa tablas Markdown cuando hay más de 2 valores comparables. Indica siempre el periodo/año de los datos. Si hay tendencias, señálalas brevemente.",
    "normativa": "Cita el artículo o resolución exacta si está en el contexto. Indica si la norma es vigente o histórica. Formato preferido: Artículo X — [contenido resumido]. Nunca inventes referencias normativas.",
    "proceso": "Usa lista numerada de pasos, clara y accionable. Incluye prerrequisitos si los hay. Indica responsable de cada paso cuando sea relevante.",
    "comparacion": "Usa tabla Markdown con columnas para cada elemento comparado. Añade una fila de 'Conclusión' o 'Recomendación' al final si aplica.",
    "sintesis": "Sintetiza los puntos más importantes en máximo 4 viñetas. Luego ofrece un párrafo de conclusión integradora.",
    "general": "Responde en prosa clara. Máximo 3 párrafos. Usa viñetas solo si hay más de 3 ítems paralelos.",
}

def route_query(query: str) -> str:
    descriptions = "\n".join(f"- {k}: {v}" for k, v in AGENT_TYPES.items())
    prompt_text = f"""Clasifica esta pregunta en exactamente una categoría:
{descriptions}
Pregunta: {query}
Responde solo con la clave (estadistica/normativa/proceso/comparacion/sintesis/general)."""
    
    if _is_qwen() or _is_deepseek():
        try:
            raw = _local_generate_full("Clasifica preguntas. Responde solo con una palabra.", prompt_text, max_new_tokens=20)
            agent = raw.strip().lower().split()[0] if raw.strip() else "general"
            return agent if agent in AGENT_TYPES else "general"
        except Exception:
            return "general"
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt_text}], temperature=0, max_tokens=20)
        agent = r.choices[0].message.content.strip().lower()
        return agent if agent in AGENT_TYPES else "general"
    except Exception:
        return "general"

def classify_intent(prompt: str, last_answer: str) -> str:
    prompt_llm = f"""Contexto — respuesta previa del sistema:
{last_answer[:400]}
Nuevo mensaje del usuario:
{prompt}
Clasifica:
"pregunta" si el usuario hace una nueva pregunta o tema
"retroalimentacion" si el usuario corrige, mejora o complementa la respuesta anterior
JSON: {{"tipo": "pregunta" o "retroalimentacion"}}"""
    
    if _is_qwen() or _is_deepseek():
        try:
            raw = _local_generate_full("Clasifica intenciones. Responde solo con JSON.", prompt_llm, max_new_tokens=50)
            data = clean_json(raw)
            return data.get("tipo", "pregunta")
        except Exception:
            return "pregunta"
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt_llm}], temperature=0, max_tokens=50)
        data = clean_json(r.choices[0].message.content)
        return data.get("tipo", "pregunta")
    except Exception:
        return "pregunta"

# ─────────────────────────────────────────────
# M6 · ANSWER AGENT v2
# ─────────────────────────────────────────────
class AnswerAgentV2:
    def _build_messages(self, query, context, memory_ctx, agent_type, sources):
        format_instr = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["general"])
        source_list = ", ".join(set(sources)) if sources else "documentos de acreditación"
        system_msg = f"""Eres ChatAcredita, asistente especializado en acreditación de la EISC, Universidad del Valle, Colombia.
REGLAS ABSOLUTAS:
Responde SOLO con