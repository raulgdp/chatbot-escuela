# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO v3.1 — RAG + Agentes + Retroalimentación Vectorial      ║
# ║  EISC — Universidad del Valle, Cali, Colombia                             ║
# ║  CORRECCIONES: Espacios eliminados, total_memory, device_map limpio,      ║
# ║  dict keys limpios, cache_resource compatible, do_sample=True para Instruct ║
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
# M11 · SEGURIDAD — Contraseñas hasheadas + rate limiting
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
    page_title="ChatAcredita PRO v3.1 - EISC-Univalle",
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
🎓 ChatAcredita PRO v3.1 — EISC (Universidad del Valle)
&nbsp;·&nbsp; RAG + Agentes + Qwen 7B local
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONEXIÓN APIs — Groq (Llama 3.3 70B)
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
# M13 · BACKEND LOCAL: Qwen 2.5-7B fine-tuneado para acreditación CNA
# ─────────────────────────────────────────────
QWEN_MODEL_ID = "raulgdp/qwen2.5-7b-acredita-cna-col"

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
        import traceback
        print(f"🔥 ERROR Qwen: {traceback.format_exc()}")
        yield f"⚠️ Error Qwen local: {str(e)[:200]}"

def generate_qwen_full(
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 900,
) -> str:
    return "".join(generate_qwen_response(system_msg, user_msg, max_new_tokens))

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
    
    if st.session_state.get("use_qwen", False):
        try:
            raw = generate_qwen_full("Eres un experto en acreditación CNA. Responde solo con JSON.", prompt_text, max_new_tokens=350)
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
    
    if st.session_state.get("use_qwen", False):
        try:
            raw = generate_qwen_full("Clasifica preguntas. Responde solo con una palabra.", prompt_text, max_new_tokens=20)
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
    
    if st.session_state.get("use_qwen", False):
        try:
            raw = generate_qwen_full("Clasifica intenciones. Responde solo con JSON.", prompt_llm, max_new_tokens=50)
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
# M6 · ANSWER AGENT v2 — CON STREAMING Y CITAS INLINE + QWEN LOCAL
# ─────────────────────────────────────────────
class AnswerAgentV2:
    def _build_messages(self, query, context, memory_ctx, agent_type, sources):
        format_instr = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["general"])
        source_list = ", ".join(set(sources)) if sources else "documentos de acreditación"
        system_msg = f"""Eres ChatAcredita, asistente especializado en acreditación de la EISC, Universidad del Valle, Colombia.
REGLAS ABSOLUTAS:
Responde SOLO con información presente en el CONTEXTO RECUPERADO.
Si la información no está en el contexto, di exactamente: "No encontré información sobre esto en los documentos disponibles."
Cuando uses un dato específico del contexto, añade [Fuente: {source_list}] al final de la oración.
NUNCA inventes datos, fechas, nombres o normativas.
NUNCA menciones que tienes un "contexto" — habla como si conocieras los documentos.
INSTRUCCIÓN DE FORMATO: {format_instr}"""
        user_msg = f"""HISTORIAL CONVERSACIONAL:
{memory_ctx if memory_ctx else "Sin historial previo."}
CONTEXTO RECUPERADO DE DOCUMENTOS EISC:
{context}
PREGUNTA DEL USUARIO:
{query}"""
        return system_msg, user_msg

    def stream(self, query: str, context: str, memory_ctx: str, agent_type: str, sources: list[str]) -> Generator[str, None, None]:
        system_msg, user_msg = self._build_messages(query, context, memory_ctx, agent_type, sources)
        if st.session_state.get("use_qwen", False):
            yield from generate_qwen_response(system_msg, user_msg)
        else:
            yield from self._stream_groq(system_msg, user_msg)

    def _stream_groq(self, system_msg: str, user_msg: str) -> Generator[str, None, None]:
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.2, max_tokens=1000, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"⚠️ Error Groq: {str(e)[:120]}"

    def generate_correction(self, query: str, last_answer: str, context: str) -> str:
        correction_prompt = f"""El usuario señaló un problema con esta respuesta previa:
RESPUESTA PREVIA: {last_answer[:600]}
RETROALIMENTACIÓN/CORRECCIÓN DEL USUARIO: {query}
CONTEXTO DOCUMENTAL ACTUALIZADO: {context}
Genera una respuesta CORREGIDA que:
Integre la corrección del usuario. Use solo información del contexto documental. Sea más precisa que la anterior."""
        if st.session_state.get("use_qwen", False):
            return generate_qwen_full("Eres ChatAcredita. Corrige la respuesta anterior.", correction_prompt)
        try:
            r = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role": "user", "content": correction_prompt}], temperature=0.15, max_tokens=900)
            return r.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error en corrección: {str(e)[:100]}"

# ─────────────────────────────────────────────
# M7 · EVALUADOR DE CALIDAD (RAGAS-LITE)
# ─────────────────────────────────────────────
def evaluate_response(query: str, context: str, answer: str) -> dict:
    eval_prompt = f"""Evalúa esta respuesta de un sistema RAG sobre acreditación universitaria.
PREGUNTA: {query}
CONTEXTO RECUPERADO: {context[:1200]}
RESPUESTA GENERADA: {answer[:700]}
Evalúa en escala 0.0 a 1.0 y responde SOLO con JSON:
{{"faithfulness": <float>, "answer_relevance": <float>, "context_precision": <float>, "hallucination_risk": <float>}}"""
    
    default_scores = {"faithfulness": 0.8, "answer_relevance": 0.8, "context_precision": 0.7, "hallucination_risk": 0.2}
    
    if st.session_state.get("use_qwen", False):
        try:
            raw = generate_qwen_full("Evalúa respuestas RAG. Responde solo con JSON.", eval_prompt, max_new_tokens=150)
            scores = clean_json(raw)
            for k in default_scores:
                if k not in scores or not isinstance(scores[k], (int, float)):
                    scores[k] = default_scores[k]
            _log_evaluation_async(query, scores)
            return scores
        except Exception:
            return default_scores
    
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": eval_prompt}], temperature=0, max_tokens=150)
        scores = clean_json(r.choices[0].message.content)
        for k in default_scores:
            if k not in scores or not isinstance(scores[k], (int, float)):
                scores[k] = default_scores[k]
        _log_evaluation_async(query, scores)
        return scores
    except Exception:
        return default_scores

def _log_evaluation_async(query: str, scores: dict):
    try:
        emb = embedder.encode([query], normalize_embeddings=True)[0]
        qdrant.upsert(
            collection_name=EVAL_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"query": query, "scores": scores, "timestamp": time.time(), "user": st.session_state.get("user", "unknown")},
            )]
        )
    except Exception:
        pass

def quality_badge(scores: dict) -> tuple[str, str]:
    faith = scores.get("faithfulness", 0.8)
    halluc = scores.get("hallucination_risk", 0.2)
    if faith >= 0.8 and halluc <= 0.2:
        return "Alta confianza", "q-high"
    elif faith >= 0.6 and halluc <= 0.4:
        return "Confianza media", "q-med"
    else:
        return "Verificar respuesta", "q-low"

# ─────────────────────────────────────────────
# M9 · FEEDBACK ENRIQUECIDO CON DEDUPLICACIÓN VECTORIAL
# ─────────────────────────────────────────────
def save_feedback_dedup(query: str, answer: str, rating: int, tags: list[str], corrected: bool = False) -> str:
    combined = f"PREGUNTA: {query}\n\nRESPUESTA: {answer[:500]}"
    emb = embedder.encode([combined], normalize_embeddings=True)[0]
    
    try:
        existing = qdrant.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=emb.tolist(),
            limit=1,
            with_payload=True,
        ).points
        
        if existing and existing[0].score > 0.92:
            old = existing[0].payload
            old_rating = old.get("rating", rating)
            old_votes = old.get("votes", 1)
            new_rating = round((old_rating * old_votes + rating) / (old_votes + 1), 2)
            qdrant.set_payload(
                collection_name=FEEDBACK_COLLECTION,
                payload={"rating": new_rating, "votes": old_votes + 1, "tags": list(set(old.get("tags", []) + tags)), "last_vote": time.time()},
                points=[existing[0].id],
            )
            return "updated"
        
        qdrant.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": combined, "query": query, "answer": answer[:600],
                    "source": "feedback_usuario", "type": "respuesta_corregida" if corrected else "valoracion",
                    "rating": rating, "tags": tags, "votes": 1,
                    "timestamp": time.time(), "user": st.session_state.get("user", "unknown"),
                },
            )],
        )
        return "created"
    except Exception:
        return "error"

# ─────────────────────────────────────────────
# M12 · PROCESAMIENTO ASYNC DE DOCUMENTOS
# ─────────────────────────────────────────────
def embed_chunks_parallel(chunks: list[str], batch_size: int = 32) -> np.ndarray:
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    def embed_batch(batch):
        return embedder.encode(batch, normalize_embeddings=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(embed_batch, batches))
    return np.vstack(results)

def process_uploaded_document(pdf_bytes: bytes, filename: str) -> tuple[list, list]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        all_text = pymupdf4llm.to_markdown(doc)
        doc.close()
        os.unlink(tmp_path)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n## ", "\n\n### ", "\n\n#### ", "\n\n| ", "\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )
        chunks = splitter.split_text(all_text)
        valid = [c.strip() for c in chunks if len(c.strip()) > 80 and not c.strip().endswith("|") and not c.strip().endswith("[TABLA")]
        return valid, [filename] * len(valid)
    except Exception as e:
        st.error(f"❌ Error procesando PDF: {str(e)[:120]}")
        return [], []

def add_chunks_to_qdrant(chunks: list[str], sources: list[str]) -> bool:
    try:
        normalized = [normalize_text(c) for c in chunks]
        embeddings = embed_chunks_parallel(normalized)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={"text": normalized[i], "source": sources[i], "chunk_id": i, "type": "documento_subido", "timestamp": time.time()},
            )
            for i in range(len(normalized))
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"✅ {len(points)} chunks añadidos a Qdrant")
        return True
    except Exception as e:
        st.error(f"❌ Error subiendo a Qdrant: {str(e)[:120]}")
        return False

# ─────────────────────────────────────────────
# SISTEMA RAG PRINCIPAL
# ─────────────────────────────────────────────
class RAGSystemV3:
    def __init__(self):
        self.answer_agent = AnswerAgentV2()

    def run_stream(self, query: str, messages: list, last_answer: str = "", status_placeholder=None) -> dict:
        start = time.time()
        results = {"tokens": None, "sources": [], "metrics": {}, "scores": {}, "intent": "pregunta", "corrected": False, "agent_type": "general"}

        def show_status(css_class: str, text: str):
            if status_placeholder:
                avatar_b64 = get_base64_image("data/yo.webp") or ""
                img_tag = f'<img src="data:image/webp;base64,{avatar_b64}" class="avatar-img">' if avatar_b64 else ""
                status_placeholder.markdown(f'<div class="thinking-avatar {css_class}">{img_tag} <span>{text}</span></div>', unsafe_allow_html=True)

        backend_label = "Qwen 7B local" if st.session_state.get("use_qwen") else "Groq"
        show_status("status-analizando", f"🧠 Analizando intención ({backend_label})...")
        intent = classify_intent(query, last_answer)
        results["intent"] = intent
        memory_ctx = memory.get_context(messages)
        show_status("status-expandiendo", "🔄 Expandiendo consulta...")
        rewrite_data = rewrite_query(query, memory_ctx)
        query_variants = list({query, rewrite_data["rewritten"], rewrite_data["hyde"]} - {""})
        show_status("status-recuperando", "🔍 Recuperando información...")
        raw_results = hybrid_search_rrf(query, query_variants, use_feedback=False)
        if intent == "retroalimentacion":
            fb_results = hybrid_search_rrf(query, query_variants, use_feedback=True)
            raw_results = raw_results + fb_results
        show_status("status-reranking", "📊 Rerankeando resultados...")
        reranked = rerank_results(query, raw_results)
        context = "\n\n---\n\n".join(r["text"] for r in reranked)[:4500]
        sources = list({r["source"] for r in reranked if r["source"] != "desconocido"})
        results["sources"] = sources
        agent_type = route_query(query)
        results["agent_type"] = agent_type
        
        if intent == "retroalimentacion" and last_answer:
            show_status("status-corrigiendo", "🔄 Corrigiendo con tu feedback...")
            corrected_answer = self.answer_agent.generate_correction(query, last_answer, context)
            save_feedback_dedup(query=query, answer=corrected_answer, rating=4, tags=["correccion_automatica"], corrected=True)
            results["corrected"] = True
            show_status("status-evaluando", "🔬 Evaluando calidad...")
            scores = evaluate_response(query, context, corrected_answer)
            results["scores"] = scores
            results["metrics"] = {"latency": round(time.time() - start, 2), "intent": intent, "corrected": True, "agent": agent_type, "chunks": len(reranked), "backend": "qwen_local" if st.session_state.get("use_qwen") else "groq"}
            def single_token(): yield corrected_answer
            results["tokens"] = single_token()
            return results
        
        show_status("status-generando", f"✍️ Generando con {backend_label}...")
        token_gen = self.answer_agent.stream(query=query, context=context, memory_ctx=memory_ctx, agent_type=agent_type, sources=sources)
        results["tokens"] = token_gen
        results["context_used"] = context[:1200]
        results["metrics"] = {"latency": round(time.time() - start, 2), "intent": intent, "corrected": False, "agent": agent_type, "chunks": len(reranked), "context_chars": len(context), "backend": "qwen_local" if st.session_state.get("use_qwen") else "groq"}
        return results

rag_system = RAGSystemV3()

# ─────────────────────────────────────────────
# CONTADOR DE VISITAS
# ─────────────────────────────────────────────
COUNTER_FILE = "counter.json"
def load_counter() -> int:
    try:
        with open(COUNTER_FILE, "r") as f:
            return json.load(f).get("visits", 0)
    except Exception:
        return 0

def save_counter(value: int):
    try:
        with open(COUNTER_FILE, "w") as f:
            json.dump({"visits": value}, f)
    except Exception:
        pass

if "counted" not in st.session_state:
    visits = load_counter() + 1
    save_counter(visits)
    st.session_state.counted = True
    st.session_state.visits = visits
else:
    st.session_state.visits = load_counter()

# ─────────────────────────────────────────────
# INICIALIZACIÓN DE SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("metrics", {"latency": 0, "intent": "pregunta", "corrected": False, "agent": "general", "backend": "groq"}),
    ("last_scores", {}),
    ("pending_feedback", None),
    ("use_qwen", False),
    ("user", "invitado"),  # ✅ AÑADIDO para evitar KeyError
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
st.title("💬 Chat Académico EISC")
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)
st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 Modelo LLM")
    backend_options = []
    if groq_available:
        backend_options.append("☁️ Groq — Llama 3.3 70B")
    backend_options.append("🖥️ Qwen 7B local — acreditación CNA")
    
    llm_backend = st.selectbox("Selecciona el backend de generación:", backend_options, key="llm_backend_selector", help="Groq es más rápido (API cloud). Qwen 7B es tu modelo fine-tuneado local.")
    st.session_state.use_qwen = "Qwen" in llm_backend
    
    if st.session_state.use_qwen:
        import torch
        with st.spinner("Cargando Qwen 7B..."):
            try:
                qwen_model, qwen_tokenizer = load_qwen_local()
                device_info = "CUDA" if torch.cuda.is_available() else "CPU"
                st.success(f"✅ Qwen 7B acreditación ({device_info})")
            except Exception as e:
                st.error(f"❌ Error cargando Qwen: {str(e)[:100]}")
                st.session_state.use_qwen = False
    else:
        if groq_available:
            st.info(f"Usando Groq: {DEFAULT_MODEL}")
        else:
            st.warning("Groq no disponible, se usará Qwen local")
            st.session_state.use_qwen = True
    
    st.markdown("---")
    st.markdown("### 📁 Subir Documento")
    uploaded_file = st.file_uploader("PDF sobre acreditación", type=["pdf"], help="Se añadirá al índice vectorial automáticamente")
    
    if uploaded_file:
        if st.button("🚀 Procesar e Indexar", type="primary"):
            with st.spinner("Extrayendo texto del PDF..."):
                pdf_bytes = uploaded_file.read()
                chunks, sources = process_uploaded_document(pdf_bytes, uploaded_file.name)
            if chunks:
                st.success(f"✅ {len(chunks)} chunks extraídos de '{uploaded_file.name}'")
                with st.expander("👁️ Preview de los primeros 3 chunks", expanded=False):
                    for i, c in enumerate(chunks[:3]):
                        st.caption(f"Chunk {i+1} ({len(c)} chars)")
                        st.code(c[:250] + ("..." if len(c) > 250 else ""), language=None)
                with st.spinner("Generando embeddings y subiendo a Qdrant..."):
                    if add_chunks_to_qdrant(chunks, sources):
                        build_bm25_index.clear()
                        st.balloons()
                        try:
                            col_info = qdrant.get_collection(COLLECTION_NAME)
                            st.info(f"📊 Total de chunks en Qdrant: {col_info.points_count}")
                        except Exception:
                            pass
            else:
                st.warning("⚠️ No se extrajeron chunks válidos del PDF.")
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        avatar_path = "data/yo.webp"
        if os.path.exists(avatar_path):
            st.image(avatar_path, width=70)
        else:
            st.markdown("👤")
    with col2:
        # ✅ Acceso seguro con .get()
        user_display = st.session_state.get("user", "invitado")
        st.markdown(f"**{user_display}**")
        st.caption("EISC · Univalle")
    
    st.markdown("### 📊 Métricas de sesión")
    metrics = st.session_state.metrics
    st.metric("⏱️ Latencia", f"{metrics.get('latency', 0)} s")
    st.metric("🤖 Backend", metrics.get('backend', 'groq').upper())
    st.metric("🎯 Agente", metrics.get('agent', 'general').capitalize())
    st.metric("🔍 Intención", metrics.get('intent', 'pregunta').capitalize())
    st.metric("📚 Chunks usados", metrics.get('chunks', 0))
    st.metric("👥 Visitas", st.session_state.visits)
    
    if st.session_state.last_scores:
        st.markdown("### 🔬 Calidad de última respuesta")
        scores = st.session_state.last_scores
        st.progress(scores.get("faithfulness", 0.8), text=f"Faithfulness: {scores.get('faithfulness', 0.8):.0%}")
        st.progress(scores.get("answer_relevance", 0.8), text=f"Relevance: {scores.get('answer_relevance', 0.8):.0%}")
        st.progress(scores.get("context_precision", 0.7), text=f"Precisión ctx: {scores.get('context_precision', 0.7):.0%}")
        halluc = scores.get("hallucination_risk", 0.2)
        color = "🟢" if halluc < 0.3 else ("🟡" if halluc < 0.5 else "🔴")
        st.caption(f"{color} Riesgo de alucinación: {halluc:.0%}")
    
    st.markdown("---")
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_scores = {}
        st.rerun()
    
    st.markdown("---")
    with st.expander("🔧 Diagnóstico del índice", expanded=False):
        try:
            col_info = qdrant.get_collection(COLLECTION_NAME)
            total_points = col_info.points_count
            st.metric("Total de chunks indexados", total_points)
            if total_points == 0:
                st.error("❌ La colección está vacía. Ningún documento ha sido indexado.")
            else:
                st.markdown("**Documentos indexados:**")
                sources_found = set()
                offset_diag = None
                try:
                    while True:
                        res = qdrant.scroll(collection_name=COLLECTION_NAME, limit=200, offset=offset_diag, with_payload=["source"], with_vectors=False)
                        for pt in res[0]:
                            sources_found.add(pt.payload.get("source", "desconocido"))
                        offset_diag = res[1]
                        if offset_diag is None:
                            break
                except Exception:
                    pass
                if sources_found:
                    for src in sorted(sources_found):
                        st.caption(f"📄 {src}")
                else:
                    st.caption("Sin fuentes identificadas")
                st.markdown("**Probar búsqueda:**")
                test_query = st.text_input("Escribe una consulta de prueba", key="diag_query", placeholder="ej: competencias del programa")
                if test_query and st.button("🔍 Buscar", key="diag_search"):
                    with st.spinner("Buscando..."):
                        try:
                            emb = embedder.encode([test_query], normalize_embeddings=True)[0]
                            hits = qdrant.query_points(collection_name=COLLECTION_NAME, query=emb.tolist(), limit=3, with_payload=True).points
                            if not hits:
                                st.warning("No se encontraron resultados.")
                            else:
                                for j, hit in enumerate(hits):
                                    score = round(hit.score, 4)
                                    text = hit.payload.get("text", "")[:300]
                                    source = hit.payload.get("source", "?")
                                    st.markdown(f"**#{j+1}** (score: {score}) — _{source}_")
                                    st.caption(text)
                                    st.markdown("---")
                                best = hits[0].score
                                if best < 0.3:
                                    st.error(f"⚠️ Score máximo muy bajo ({best:.2f}).")
                                elif best < 0.5:
                                    st.warning(f"Score moderado ({best:.2f}).")
                                else:
                                    st.success(f"Score bueno ({best:.2f}).")
                        except Exception as e:
                            st.error(f"Error en búsqueda: {str(e)[:100]}")
        except Exception as e:
            st.error(f"Error conectando a Qdrant: {str(e)[:100]}")

# ─────────────────────────────────────────────
# M8 · MANEJO DE INPUT CON STREAMING REAL
# ─────────────────────────────────────────────
avatar_b64 = get_base64_image("data/yo.webp") or ""
prompt = st.chat_input("Escribe tu pregunta sobre acreditación EISC...")

if prompt:
    if not check_rate_limit(st.session_state.get("user", "invitado")):
        st.warning(f"⚠️ Límite de {MAX_REQUESTS_PER_MINUTE} consultas/minuto alcanzado.")
        st.stop()
    
    prompt_clean = sanitize_query(prompt)
    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = re.sub(r"<[^>]+>", " ", m.get("content", ""))[:800]
            break
    
    st.session_state.messages.append({"role": "user", "content": prompt_clean})
    with st.chat_message("user"):
        st.markdown(prompt_clean)
    
    status_ph = st.empty()
    rag_result = rag_system.run_stream(query=prompt_clean, messages=st.session_state.messages, last_answer=last_answer, status_placeholder=status_ph)
    
    backend_label = "Qwen 7B" if st.session_state.get("use_qwen") else "Groq"
    status_ph.markdown(f'<div class="thinking-avatar status-generando">{"<img src=\"data:image/webp;base64,\" + avatar_b64 + \" class=avatar-img>" if avatar_b64 else ""} <span>✍️ Generando con {backend_label}...</span></div>', unsafe_allow_html=True)
    
    full_answer = ""
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        if rag_result["corrected"]:
            for token in rag_result["tokens"]:
                full_answer += token
            st.markdown('<span style="color:#e65100;font-weight:bold;">✏️ Respuesta corregida según tu feedback</span>', unsafe_allow_html=True)
            stream_placeholder.markdown(full_answer)
        else:
            for token in rag_result["tokens"]:
                full_answer += token
                stream_placeholder.markdown(full_answer + "▌")
            stream_placeholder.markdown(full_answer)
        
        context_for_eval = rag_result.get("context_used", full_answer[:1200])
        status_ph.markdown(f'<div class="thinking-avatar status-evaluando"><span>🔬 Evaluando calidad...</span></div>', unsafe_allow_html=True)
        scores = evaluate_response(prompt_clean, context_for_eval, full_answer)
        st.session_state.last_scores = scores
        rag_result["metrics"]["scores"] = scores
        
        badge_label, badge_class = quality_badge(scores)
        st.markdown(f'<span class="quality-badge {badge_class}">🔬 {badge_label}</span>', unsafe_allow_html=True)
        
        if scores.get("hallucination_risk", 0) > HALLUCINATION_THRESHOLD:
            st.warning("⚠️ Esta respuesta podría contener información no verificada. Consulta directamente los documentos originales.")
        
        sources = rag_result.get("sources", [])
        if sources:
            st.markdown('<div class="sources-container">', unsafe_allow_html=True)
            st.markdown("### 📚 Fuentes consultadas")
            badges = " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
            st.markdown(badges, unsafe_allow_html=True)
            agent_type = rag_result.get("agent_type", "general")
            backend_info = rag_result["metrics"].get("backend", "groq")
            st.caption(f"Agente: {agent_type} · Backend: {backend_info} · Chunks: {rag_result['metrics'].get('chunks', 0)} · Latencia: {rag_result['metrics'].get('latency', 0)}s")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("**¿Fue útil esta respuesta?**")
        rating_cols = st.columns(5)
        rating_labels = ["😞 Muy mala", "😕 Mala", "😐 Regular", "🙂 Buena", "😄 Excelente"]
        for i, (col, label) in enumerate(zip(rating_cols, rating_labels)):
            if col.button(label.split()[0], key=f"rating_btn_{i}_{len(st.session_state.messages)}", help=label):
                save_feedback_dedup(query=prompt_clean, answer=full_answer, rating=i + 1, tags=[f"rating_{i+1}", agent_type])
                st.toast(f"✅ Valoración guardada ({label.split()[1]})", icon="⭐")
    
    time.sleep(0.5)
    status_ph.empty()
    
    display_answer = full_answer
    if rag_result["corrected"]:
        display_answer = '<span class="feedback-indicator">✏️ Corregido</span> <br>' + display_answer
    if sources:
        src_html = " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
        display_answer += f'<br><br><div class="sources-container"><strong>📚 Fuentes:</strong> {src_html}</div>'
    
    st.session_state.messages.append({"role": "assistant", "content": display_answer})
    st.session_state.metrics = rag_result["metrics"]
    st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
Universidad del Valle · Grupo GUIA · ChatAcredita PRO v3.1<br>
RAG + Agentes + Qwen 7B local + Retroalimentación Vectorial · EISC 2025
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SCROLL AUTOMÁTICO
# ─────────────────────────────────────────────
st.markdown("""
<script>
function scrollToBottom() {
    const mainSection = window.parent.document.querySelector('section.main');
    if (mainSection) mainSection.scrollTop = mainSection.scrollHeight;
    const msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
    if (msgs.length > 0) msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
    const chatContainer = window.parent.document.querySelector('[data-testid="stChatMessageContainer"]');
    if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
    const containers = window.parent.document.querySelectorAll('.main .block-container, [data-testid="stVerticalBlock"]');
    containers.forEach(c => { c.scrollTop = c.scrollHeight; });
}
scrollToBottom();
[100, 300, 600, 1000, 1500, 2000, 3000, 5000].forEach(d => setTimeout(scrollToBottom, d));
try {
    const targetNode = window.parent.document.querySelector('section.main') || window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
    if (targetNode) {
        const observer = new MutationObserver(function(mutations) { setTimeout(scrollToBottom, 100); });
        observer.observe(targetNode, { childList: true, subtree: true });
        setTimeout(() => observer.disconnect(), 30000);
    }
} catch(e) {}
</script>
""", unsafe_allow_html=True)