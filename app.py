# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO v3.0 — RAG + Agentes + Retroalimentación Vectorial        ║
# ║  EISC — Universidad del Valle, Cali, Colombia                               ║
# ║  Mejoras: M1 Query Rewriting · M2 BM25 Real · M3 Reranker · M4 Memoria     ║
# ║           M5 Router Agentes · M6 AnswerAgent v2 · M7 Evaluador RAGAS        ║
# ║           M8 Streaming · M9 Feedback Deduplicado · M10 Observabilidad       ║
# ║           M11 Seguridad · M12 Procesamiento Async                           ║
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

# ══════════════════════════════════════════════════════════════════════════════
# M11 · SEGURIDAD — Contraseñas hasheadas + rate limiting
# ══════════════════════════════════════════════════════════════════════════════
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
    """Elimina intentos de prompt injection y limita longitud."""
    for pattern in INJECTION_PATTERNS:
        query = re.sub(pattern, "[eliminado]", query, flags=re.IGNORECASE)
    return query[:MAX_QUERY_LENGTH].strip()

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatAcredita PRO v3 - EISC-Univalle",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLLECTION_NAME        = "acreditacion"
FEEDBACK_COLLECTION    = "feedback_acreditacion"
EVAL_COLLECTION        = "evaluaciones_chatacredita"
TOP_K                  = 8   # Aumentado de 5 a 8 para reranking posterior
TOP_K_FINAL            = 5   # Después del reranker
HALLUCINATION_THRESHOLD = 0.4  # Score máximo tolerable de riesgo de alucinación

# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# CSS + HEADER
# ══════════════════════════════════════════════════════════════════════════════
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
.status-expandiendo { background:#ede7f6; border-left:3px solid #7e57c2; color:#4527a0; }
.status-recuperando { background:#e8f5e9; border-left:3px solid #4caf50; color:#2e7d32; }
.status-reranking   { background:#fff8e1; border-left:3px solid #ffc107; color:#f57f17; }
.status-generando   { background:#f3e5f5; border-left:3px solid #9c27b0; color:#4a148c; }
.status-evaluando   { background:#fce4ec; border-left:3px solid #e91e63; color:#880e4f; }
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
    70%  { box-shadow: 0 0 0 8px rgba(255,152,0,0);   }
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
.quality-badge {
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
    🎓 ChatAcredita PRO v3 — EISC (Universidad del Valle)
    &nbsp;·&nbsp; RAG + Agentes + Retroalimentación Vectorial
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONEXIÓN APIs
# ══════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY  = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL   = "anthropic/claude-opus-4.6-fast"
FAST_MODEL      = "openai/gpt-4o-mini"  # Para clasificación rápida

try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    _ = client.models.list()
    st.sidebar.success(f"✅ OpenRouter: {DEFAULT_MODEL}")
except Exception as e:
    st.sidebar.error(f"❌ OpenRouter: {str(e)[:80]}")
    st.sidebar.info("Verifica OPENAI_API_KEY en Secrets")

try:
    qdrant = QdrantClient(
        url=get_secret("QDRANT_URL", "").strip(),
        api_key=get_secret("QDRANT_API_KEY", "").strip(),
    )
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in existing:
        st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada")
        st.stop()

    # Crear colecciones auxiliares si no existen
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

# ══════════════════════════════════════════════════════════════════════════════
# MODELOS DE EMBEDDINGS + RERANKER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3", device="cpu")

@st.cache_resource
def load_reranker():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None  # Reranker es opcional

embedder = load_embedder()
reranker = load_reranker()
st.sidebar.success("✅ Embeddings: BGE-M3 (1024d)")
if reranker:
    st.sidebar.success("✅ Reranker: ms-marco-MiniLM")

# ══════════════════════════════════════════════════════════════════════════════
# M4 · MEMORIA CONVERSACIONAL
# ══════════════════════════════════════════════════════════════════════════════
class ConversationMemory:
    """Gestiona contexto conversacional con compresión progresiva."""

    def __init__(self, max_turns: int = 6, max_summary_tokens: int = 150):
        self.max_turns = max_turns
        self.max_summary_tokens = max_summary_tokens

    def get_context(self, messages: list) -> str:
        clean = [m for m in messages if m.get("role") in ("user", "assistant")]
        recent = clean[-(self.max_turns * 2):]

        if len(clean) <= self.max_turns * 2:
            return self._format(recent)

        # Comprimir turnos antiguos
        old = clean[:-(self.max_turns * 2)]
        summary = self._summarize(old)
        return f"[Resumen diálogo anterior]: {summary}\n\n" + self._format(recent)

    def _summarize(self, messages: list) -> str:
        dialog = self._format(messages)
        try:
            r = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Resume en 2 oraciones este diálogo sobre acreditación EISC:\n{dialog}",
                }],
                temperature=0,
                max_tokens=self.max_summary_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return "[historial previo disponible]"

    def _format(self, messages: list) -> str:
        lines = []
        for m in messages:
            role = "Usuario" if m["role"] == "user" else "Asistente"
            content = re.sub(r"<[^>]+>", "", m.get("content", ""))[:400]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

memory = ConversationMemory()

# ══════════════════════════════════════════════════════════════════════════════
# M1 · QUERY REWRITING + HyDE
# ══════════════════════════════════════════════════════════════════════════════
def rewrite_query(query: str, memory_ctx: str) -> dict:
    """
    Genera variantes semánticas de la query para mejorar el recall.
    Retorna: rewritten, hyde (doc hipotético), keywords, lang.
    """
    prompt = f"""Eres un experto en acreditación universitaria colombiana (CNA).

Contexto conversacional reciente:
{memory_ctx[-600:] if memory_ctx else 'Sin historial previo.'}

Query del usuario: {query}

Genera un JSON con:
- "rewritten": reformulación más precisa y técnica de la query (string)
- "hyde": párrafo de 2-3 oraciones que podría aparecer en un documento de acreditación respondiendo esta query (string)
- "keywords": lista de 4-6 términos clave para búsqueda BM25 (list)
- "lang": idioma detectado "es" o "en" (string)

Solo JSON sin markdown."""

    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=350,
        )
        data = clean_json(r.choices[0].message.content)
        return {
            "rewritten": data.get("rewritten", query),
            "hyde":      data.get("hyde", ""),
            "keywords":  data.get("keywords", []),
            "lang":      data.get("lang", "es"),
        }
    except Exception:
        return {"rewritten": query, "hyde": "", "keywords": [], "lang": "es"}

# ══════════════════════════════════════════════════════════════════════════════
# M2 · ÍNDICE BM25 REAL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(ttl=3600)
def build_bm25_index() -> tuple:
    """Carga todos los chunks de Qdrant y construye el índice BM25.
    Usa text_norm (normalizado) para el índice BM25 y text (original) para display."""
    all_texts, all_ids, all_sources = [], [], []
    offset = None

    try:
        while True:
            result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in result[0]:
                if point.payload and point.payload.get("text"):
                    # M12.1: usar texto original para display
                    all_texts.append(point.payload["text"])
                    all_ids.append(point.id)
                    all_sources.append(point.payload.get("source", "desconocido"))
            offset = result[1]
            if offset is None:
                break

        # M12.1: tokenizar con texto normalizado para mejor recall BM25
        tokenized = [normalize_text(t).split() for t in all_texts]
        bm25 = BM25Okapi(tokenized)
        return bm25, all_texts, all_ids, all_sources
    except Exception:
        return None, [], [], []

# ══════════════════════════════════════════════════════════════════════════════
# M2 + M3 · BÚSQUEDA HÍBRIDA CON RRF + RERANKER
# ══════════════════════════════════════════════════════════════════════════════
def hybrid_search_rrf(
    query: str,
    query_variants: list[str],
    use_feedback: bool = False,
    k_rrf: int = 60,
) -> list[dict]:
    """
    Búsqueda híbrida: Dense (BGE-M3) + BM25 fusionados con RRF.
    Retorna lista de {text, source, score, id}.
    """
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    rrf_scores: dict[str, float] = {}
    id_to_payload: dict[str, dict] = {}

    # 1. Dense search por cada variante de query
    for q in query_variants:
        try:
            emb = embedder.encode([q], normalize_embeddings=True)[0]
            results = qdrant.query_points(
                collection_name=collection,
                query=emb.tolist(),
                limit=TOP_K,
                with_payload=True,
            ).points

            for rank, r in enumerate(results):
                pid = str(r.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
                if r.payload:
                    id_to_payload[pid] = r.payload
        except Exception:
            pass

    # 2. BM25 search (solo para colección principal)
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
                        id_to_payload[pid] = {
                            "text":   bm25_texts[idx],
                            "source": bm25_sources[idx],
                        }

    # 3. Ordenar por RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    results = []
    for pid, score in sorted_ids:
        payload = id_to_payload.get(pid, {})
        if payload.get("text"):
            results.append({
                "id":     pid,
                "text":   payload["text"],
                "source": payload.get("source", "desconocido"),
                "score":  round(score, 4),
            })

    return results

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    """M3: Reordena con cross-encoder si está disponible."""
    if reranker is None or not results:
        return results[:TOP_K_FINAL]

    try:
        pairs = [(query, r["text"]) for r in results]
        scores = reranker.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:TOP_K_FINAL]
    except Exception:
        return results[:TOP_K_FINAL]

# ══════════════════════════════════════════════════════════════════════════════
# M5 · ROUTER DE AGENTES
# ══════════════════════════════════════════════════════════════════════════════
AGENT_TYPES = {
    "estadistica":  "Pregunta sobre números, tasas, porcentajes, cantidades, rankings",
    "normativa":    "Pregunta sobre reglamentos, resoluciones, leyes, artículos, normas CNA",
    "proceso":      "Pregunta sobre pasos, procedimientos, flujos, cómo hacer algo",
    "comparacion":  "Pregunta que compara dos o más elementos, criterios o periodos",
    "sintesis":     "Pregunta de resumen, conclusión o visión general de múltiples aspectos",
    "general":      "Cualquier otra pregunta sobre acreditación EISC",
}

AGENT_PROMPTS = {
    "estadistica": (
        "Presenta los datos numéricos con precisión. "
        "Usa tablas Markdown cuando hay más de 2 valores comparables. "
        "Indica siempre el periodo/año de los datos. "
        "Si hay tendencias, señálalas brevemente."
    ),
    "normativa": (
        "Cita el artículo o resolución exacta si está en el contexto. "
        "Indica si la norma es vigente o histórica. "
        "Formato preferido: **Artículo X** — [contenido resumido]. "
        "Nunca inventes referencias normativas."
    ),
    "proceso": (
        "Usa lista numerada de pasos, clara y accionable. "
        "Incluye prerrequisitos si los hay. "
        "Indica responsable de cada paso cuando sea relevante."
    ),
    "comparacion": (
        "Usa tabla Markdown con columnas para cada elemento comparado. "
        "Añade una fila de 'Conclusión' o 'Recomendación' al final si aplica."
    ),
    "sintesis": (
        "Sintetiza los puntos más importantes en máximo 4 viñetas. "
        "Luego ofrece un párrafo de conclusión integradora."
    ),
    "general": (
        "Responde en prosa clara. Máximo 3 párrafos. "
        "Usa viñetas solo si hay más de 3 ítems paralelos."
    ),
}

def route_query(query: str) -> str:
    """Clasifica la query para despachar al agente correcto."""
    descriptions = "\n".join(f"- {k}: {v}" for k, v in AGENT_TYPES.items())
    prompt = f"""Clasifica esta pregunta en exactamente una categoría:
{descriptions}

Pregunta: {query}

Responde solo con la clave (estadistica/normativa/proceso/comparacion/sintesis/general)."""

    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        agent = r.choices[0].message.content.strip().lower()
        return agent if agent in AGENT_TYPES else "general"
    except Exception:
        return "general"

def classify_intent(prompt: str, last_answer: str) -> str:
    """Distingue pregunta nueva vs retroalimentación/corrección."""
    prompt_llm = f"""Contexto — respuesta previa del sistema:
{last_answer[:400]}

Nuevo mensaje del usuario:
{prompt}

Clasifica:
- "pregunta" si el usuario hace una nueva pregunta o tema
- "retroalimentacion" si el usuario corrige, mejora o complementa la respuesta anterior

JSON: {{"tipo": "pregunta" o "retroalimentacion"}}"""

    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt_llm}],
            temperature=0,
            max_tokens=50,
        )
        data = clean_json(r.choices[0].message.content)
        return data.get("tipo", "pregunta")
    except Exception:
        return "pregunta"

# ══════════════════════════════════════════════════════════════════════════════
# M6 · ANSWER AGENT v2 — CON STREAMING Y CITAS INLINE
# ══════════════════════════════════════════════════════════════════════════════
class AnswerAgentV2:
    """
    Genera respuestas con:
    - Prompt dinámico según tipo de agente
    - Citas inline de fuentes
    - Instrucción de no-alucinación explícita
    - Streaming token a token
    """

    def stream(
        self,
        query: str,
        context: str,
        memory_ctx: str,
        agent_type: str,
        sources: list[str],
    ) -> Generator[str, None, None]:

        format_instr = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["general"])
        source_list  = ", ".join(set(sources)) if sources else "documentos de acreditación"

        system_msg = f"""Eres ChatAcredita, asistente especializado en acreditación de la EISC, Universidad del Valle, Colombia.

REGLAS ABSOLUTAS:
1. Responde SOLO con información presente en el CONTEXTO RECUPERADO.
2. Si la información no está en el contexto, di exactamente: "No encontré información sobre esto en los documentos disponibles."
3. Cuando uses un dato específico del contexto, añade [Fuente: {source_list}] al final de la oración.
4. NUNCA inventes datos, fechas, nombres o normativas.
5. NUNCA menciones que tienes un "contexto" — habla como si conocieras los documentos.

INSTRUCCIÓN DE FORMATO: {format_instr}"""

        user_msg = f"""HISTORIAL CONVERSACIONAL:
{memory_ctx if memory_ctx else "Sin historial previo."}

CONTEXTO RECUPERADO DE DOCUMENTOS EISC:
{context}

PREGUNTA DEL USUARIO:
{query}"""

        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=1000,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"⚠️ Error al generar respuesta: {str(e)[:120]}"

    def generate_correction(
        self, query: str, last_answer: str, context: str
    ) -> str:
        """Genera respuesta corregida basada en retroalimentación del usuario."""
        prompt = f"""El usuario señaló un problema con esta respuesta previa:

RESPUESTA PREVIA:
{last_answer[:600]}

RETROALIMENTACIÓN/CORRECCIÓN DEL USUARIO:
{query}

CONTEXTO DOCUMENTAL ACTUALIZADO:
{context}

Genera una respuesta CORREGIDA que:
1. Integre la corrección del usuario
2. Use solo información del contexto documental
3. Sea más precisa que la anterior
4. Mantenga el mismo formato que la respuesta original"""

        try:
            r = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
                max_tokens=900,
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error en corrección: {str(e)[:100]}"

# ══════════════════════════════════════════════════════════════════════════════
# M7 · EVALUADOR DE CALIDAD (RAGAS-LITE)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_response(query: str, context: str, answer: str) -> dict:
    """
    Evalúa faithfulness, relevance y riesgo de alucinación.
    Persiste resultados en Qdrant para análisis posterior.
    """
    eval_prompt = f"""Evalúa esta respuesta de un sistema RAG sobre acreditación universitaria.

PREGUNTA: {query}
CONTEXTO RECUPERADO: {context[:1200]}
RESPUESTA GENERADA: {answer[:700]}

Evalúa en escala 0.0 a 1.0 y responde SOLO con JSON:
{{
  "faithfulness": <float>,
  "answer_relevance": <float>,
  "context_precision": <float>,
  "hallucination_risk": <float>
}}

Definiciones:
- faithfulness: fracción de afirmaciones de la respuesta que están en el contexto (1.0=todo, 0.0=nada)
- answer_relevance: qué tan bien responde la pregunta (1.0=perfecta, 0.0=irrelevante)
- context_precision: qué tan relevante era el contexto para la pregunta (1.0=perfecto, 0.0=irrelevante)
- hallucination_risk: probabilidad de que la respuesta contenga info inventada (0.0=ninguna, 1.0=alta)"""

    default_scores = {
        "faithfulness":      0.8,
        "answer_relevance":  0.8,
        "context_precision": 0.7,
        "hallucination_risk": 0.2,
    }

    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            max_tokens=150,
        )
        scores = clean_json(r.choices[0].message.content)
        # Validar que todos los campos existen y son floats
        for k in default_scores:
            if k not in scores or not isinstance(scores[k], (int, float)):
                scores[k] = default_scores[k]

        # M10: Persistir evaluación en Qdrant para análisis posterior
        _log_evaluation_async(query, scores)
        return scores
    except Exception:
        return default_scores

def _log_evaluation_async(query: str, scores: dict):
    """Guarda métricas de evaluación en Qdrant (no bloquea la UI)."""
    try:
        emb = embedder.encode([query], normalize_embeddings=True)[0]
        qdrant.upsert(
            collection_name=EVAL_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "query":     query,
                    "scores":    scores,
                    "timestamp": time.time(),
                    "user":      st.session_state.get("user", "unknown"),
                },
            )],
        )
    except Exception:
        pass  # No interrumpir si falla el logging

def quality_badge(scores: dict) -> tuple[str, str]:
    """Genera badge de calidad basado en los scores de evaluación."""
    faith = scores.get("faithfulness", 0.8)
    halluc = scores.get("hallucination_risk", 0.2)

    if faith >= 0.8 and halluc <= 0.2:
        return "Alta confianza", "q-high"
    elif faith >= 0.6 and halluc <= 0.4:
        return "Confianza media", "q-med"
    else:
        return "Verificar respuesta", "q-low"

# ══════════════════════════════════════════════════════════════════════════════
# M9 · FEEDBACK ENRIQUECIDO CON DEDUPLICACIÓN VECTORIAL
# ══════════════════════════════════════════════════════════════════════════════
def save_feedback_dedup(
    query: str,
    answer: str,
    rating: int,
    tags: list[str],
    corrected: bool = False,
) -> str:
    """
    Guarda o actualiza feedback.
    Si existe un punto muy similar (cosine > 0.92), actualiza el rating promedio.
    """
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
            # Actualizar rating promedio y contador de votos
            old = existing[0].payload
            old_rating = old.get("rating", rating)
            old_votes  = old.get("votes", 1)
            new_rating = round((old_rating * old_votes + rating) / (old_votes + 1), 2)
            qdrant.set_payload(
                collection_name=FEEDBACK_COLLECTION,
                payload={
                    "rating":    new_rating,
                    "votes":     old_votes + 1,
                    "tags":      list(set(old.get("tags", []) + tags)),
                    "last_vote": time.time(),
                },
                points=[existing[0].id],
            )
            return "updated"

        # Crear nuevo punto de feedback
        qdrant.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text":      combined,
                    "query":     query,
                    "answer":    answer[:600],
                    "source":    "feedback_usuario",
                    "type":      "respuesta_corregida" if corrected else "valoracion",
                    "rating":    rating,
                    "tags":      tags,
                    "votes":     1,
                    "timestamp": time.time(),
                    "user":      st.session_state.get("user", "unknown"),
                },
            )],
        )
        return "created"
    except Exception:
        return "error"

# ══════════════════════════════════════════════════════════════════════════════
# M12 · PROCESAMIENTO DE DOCUMENTOS — MEJORADO
# ══════════════════════════════════════════════════════════════════════════════
# Mejoras:
#   M12.1  Texto original preservado en payload (legible) + normalizado solo
#          para embedding/BM25 → respuestas con acentos y formato correcto
#   M12.2  Deduplicación por hash SHA-256 del contenido del chunk →
#          evita duplicados si se sube el mismo PDF dos veces
#   M12.3  Detección de duplicado a nivel de archivo completo (hash del PDF)
#   M12.4  Tablas protegidas como chunks completos → no se cortan a la mitad
#   M12.5  Metadata enriquecida: número de página, sección/encabezado padre,
#          hash del documento origen → trazabilidad y filtrado
#   M12.6  Barra de progreso durante la indexación
#   M12.7  Estadísticas post-ingesta visibles en el sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _hash_content(content: str) -> str:
    """Genera hash SHA-256 del contenido para deduplicación."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _hash_pdf(pdf_bytes: bytes) -> str:
    """Genera hash SHA-256 del archivo PDF completo."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def _check_pdf_already_indexed(pdf_hash: str) -> bool:
    """Verifica si un PDF con este hash ya fue indexado en Qdrant."""
    try:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={
                "must": [{"key": "doc_hash", "match": {"value": pdf_hash}}]
            },
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result[0]) > 0
    except Exception:
        return False


def _get_existing_chunk_hashes() -> set[str]:
    """Recupera todos los chunk_hash existentes en Qdrant para deduplicación."""
    hashes = set()
    offset = None
    try:
        while True:
            result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=["chunk_hash"],
                with_vectors=False,
            )
            for point in result[0]:
                h = point.payload.get("chunk_hash")
                if h:
                    hashes.add(h)
            offset = result[1]
            if offset is None:
                break
    except Exception:
        pass
    return hashes


def embed_chunks_parallel(chunks: list[str], batch_size: int = 32) -> np.ndarray:
    """Genera embeddings en batches paralelos usando ThreadPoolExecutor."""
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    def embed_batch(batch):
        return embedder.encode(batch, normalize_embeddings=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(embed_batch, batches))

    return np.vstack(results)


def _extract_tables_and_text(markdown_text: str) -> list[dict]:
    """
    M12.4 — Separa el markdown en bloques de texto y tablas completas.
    Las tablas se mantienen íntegras como un solo chunk.
    Retorna lista de {"content": str, "type": "text"|"table", "section": str}
    """
    blocks = []
    current_section = "General"
    lines = markdown_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detectar encabezados para tracking de sección
        if line.startswith("## "):
            current_section = line.lstrip("# ").strip()[:80]
        elif line.startswith("### "):
            current_section = line.lstrip("# ").strip()[:80]

        # Detectar inicio de tabla (línea que empieza con |)
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            table_text = "\n".join(table_lines).strip()
            if len(table_text) > 40:  # Tabla con contenido real
                blocks.append({
                    "content": table_text,
                    "type":    "table",
                    "section": current_section,
                })
            continue

        # Texto normal: acumular hasta encontrar tabla o encabezado
        text_lines = []
        while i < len(lines) and not lines[i].strip().startswith("|"):
            text_lines.append(lines[i])
            # Si encontramos un nuevo encabezado, lo procesamos en la próxima iteración
            if lines[i].startswith("## ") or lines[i].startswith("### "):
                current_section = lines[i].lstrip("# ").strip()[:80]
            i += 1
        text_block = "\n".join(text_lines).strip()
        if len(text_block) > 30:
            blocks.append({
                "content": text_block,
                "type":    "text",
                "section": current_section,
            })

    return blocks


def _extract_page_mapping(doc: fitz.Document) -> dict[str, int]:
    """
    M12.5 — Crea un mapeo aproximado de texto a número de página.
    Retorna dict con los primeros 60 chars de cada página como clave.
    """
    page_map = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")[:120].strip()
        if text:
            # Guardar los primeros N caracteres como clave
            key = text[:60].lower().replace("\n", " ")
            page_map[key] = page_num + 1
    return page_map


def _find_page_for_chunk(chunk_text: str, page_map: dict[str, int]) -> int:
    """Busca la página más probable para un chunk dado."""
    chunk_start = chunk_text[:60].lower().replace("\n", " ")
    best_page = 0
    best_overlap = 0
    for key, page_num in page_map.items():
        # Calcular overlap de caracteres
        overlap = sum(1 for a, b in zip(chunk_start, key) if a == b)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = page_num
    return best_page if best_overlap > 15 else 0


def process_uploaded_document(
    pdf_bytes: bytes, filename: str
) -> tuple[list[dict], str]:
    """
    Procesa PDF con PyMuPDF4LLM y chunking estructural mejorado.

    Retorna:
        - Lista de dicts con {text, text_normalized, section, page, type, chunk_hash}
        - Hash del documento PDF
    """
    try:
        pdf_hash = _hash_pdf(pdf_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        page_map = _extract_page_mapping(doc)
        all_text = pymupdf4llm.to_markdown(doc)
        doc.close()
        os.unlink(tmp_path)

        # M12.4: Separar tablas como bloques completos
        blocks = _extract_tables_and_text(all_text)

        # Splitter para bloques de texto (las tablas no se fragmentan)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=[
                "\n\n## ", "\n\n### ", "\n\n#### ",
                "\n\n", "\n", ". ", " ", "",
            ],
            is_separator_regex=False,
        )

        processed_chunks = []

        for block in blocks:
            if block["type"] == "table":
                # Tablas: mantener íntegras si son menores a 2000 chars
                if len(block["content"]) <= 2000:
                    chunk_text = block["content"]
                    processed_chunks.append({
                        "text":            chunk_text,
                        "text_normalized": normalize_text(chunk_text),
                        "section":         block["section"],
                        "page":            _find_page_for_chunk(chunk_text, page_map),
                        "type":            "table",
                        "chunk_hash":      _hash_content(chunk_text),
                    })
                else:
                    # Tablas muy largas: dividir por filas en grupos
                    rows = block["content"].split("\n")
                    header = rows[0] if rows else ""
                    separator = rows[1] if len(rows) > 1 and "---" in rows[1] else ""
                    data_rows = rows[2:] if separator else rows[1:]

                    # Agrupar en bloques de ~15 filas
                    for j in range(0, len(data_rows), 15):
                        group = data_rows[j:j + 15]
                        chunk_text = "\n".join([header, separator] + group).strip()
                        if len(chunk_text) > 60:
                            processed_chunks.append({
                                "text":            chunk_text,
                                "text_normalized": normalize_text(chunk_text),
                                "section":         block["section"],
                                "page":            _find_page_for_chunk(chunk_text, page_map),
                                "type":            "table",
                                "chunk_hash":      _hash_content(chunk_text),
                            })
            else:
                # Texto: fragmentar con el splitter
                sub_chunks = splitter.split_text(block["content"])
                for sc in sub_chunks:
                    sc = sc.strip()
                    if len(sc) > 80:
                        processed_chunks.append({
                            "text":            sc,
                            "text_normalized": normalize_text(sc),
                            "section":         block["section"],
                            "page":            _find_page_for_chunk(sc, page_map),
                            "type":            "text",
                            "chunk_hash":      _hash_content(sc),
                        })

        return processed_chunks, pdf_hash

    except Exception as e:
        st.error(f"❌ Error procesando PDF: {str(e)[:120]}")
        return [], ""


def add_chunks_to_qdrant(
    chunks: list[dict],
    filename: str,
    pdf_hash: str,
    progress_bar=None,
) -> dict:
    """
    Sube chunks a Qdrant con:
    - M12.1: texto original preservado + normalizado separado
    - M12.2: deduplicación por hash de chunk
    - M12.5: metadata enriquecida
    - M12.6: barra de progreso

    Retorna estadísticas: {total, nuevos, duplicados, errores}
    """
    stats = {"total": len(chunks), "nuevos": 0, "duplicados": 0, "errores": 0}

    if not chunks:
        return stats

    # M12.2: Obtener hashes existentes para deduplicar
    existing_hashes = _get_existing_chunk_hashes()

    # Filtrar duplicados antes de hacer embedding (ahorra cómputo)
    new_chunks = []
    for c in chunks:
        if c["chunk_hash"] in existing_hashes:
            stats["duplicados"] += 1
        else:
            new_chunks.append(c)

    if not new_chunks:
        return stats

    try:
        # Generar embeddings solo del texto normalizado (para búsqueda)
        texts_for_embedding = [c["text_normalized"] for c in new_chunks]
        embeddings = embed_chunks_parallel(texts_for_embedding)

        # Construir puntos con metadata enriquecida
        points = []
        for i, chunk in enumerate(new_chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload={
                        # M12.1: texto legible para respuestas
                        "text":       chunk["text"],
                        # M12.1: texto normalizado para BM25
                        "text_norm":  chunk["text_normalized"],
                        # M12.5: metadata enriquecida
                        "source":     filename,
                        "doc_hash":   pdf_hash,
                        "chunk_hash": chunk["chunk_hash"],
                        "chunk_id":   i,
                        "section":    chunk.get("section", ""),
                        "page":       chunk.get("page", 0),
                        "chunk_type": chunk.get("type", "text"),
                        "type":       "documento_subido",
                        "timestamp":  time.time(),
                        "user":       st.session_state.get("user", "unknown"),
                    },
                )
            )

        # M12.6: Subir en batches con progreso
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            stats["nuevos"] += len(batch)
            if progress_bar:
                progress_bar.progress(
                    min((i + batch_size) / len(points), 1.0),
                    text=f"Indexando... {min(i + batch_size, len(points))}/{len(points)} chunks",
                )

        return stats

    except Exception as e:
        stats["errores"] = len(new_chunks) - stats["nuevos"]
        st.error(f"❌ Error subiendo a Qdrant: {str(e)[:120]}")
        return stats
        
# ══════════════════════════════════════════════════════════════════════════════
# SISTEMA RAG PRINCIPAL — INTEGRA TODAS LAS MEJORAS
# ══════════════════════════════════════════════════════════════════════════════
class RAGSystemV3:
    """
    Pipeline RAG completo con:
    M1 Query rewriting + HyDE
    M2 BM25 real + Dense (RRF)
    M3 Cross-encoder reranker
    M4 Memoria conversacional
    M5 Router de agentes
    M6 AnswerAgent v2 con streaming
    M7 Evaluador RAGAS-lite
    M9 Feedback deduplicado
    """

    def __init__(self):
        self.answer_agent = AnswerAgentV2()

    def run_stream(
        self,
        query: str,
        messages: list,
        last_answer: str = "",
        status_placeholder=None,
    ) -> dict:
        """
        Ejecuta el pipeline completo.
        Retorna dict con: tokens_generator, sources, metrics, scores, intent.
        """
        start = time.time()
        results = {
            "tokens":    None,
            "sources":   [],
            "metrics":   {},
            "scores":    {},
            "intent":    "pregunta",
            "corrected": False,
            "agent_type": "general",
        }

        def show_status(css_class: str, text: str):
            if status_placeholder:
                avatar_b64 = get_base64_image("data/yo.webp") or ""
                img_tag = f'<img src="data:image/webp;base64,{avatar_b64}" class="avatar-img">' if avatar_b64 else ""
                status_placeholder.markdown(
                    f'<div class="thinking-avatar {css_class}">'
                    f'{img_tag}<span>{text}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── ETAPA 1: Clasificar intención ──────────────────────────────────
        show_status("status-analizando", "🧠 Analizando intención...")
        intent = classify_intent(query, last_answer)
        results["intent"] = intent

        # ── ETAPA 2: Memoria conversacional ───────────────────────────────
        memory_ctx = memory.get_context(messages)

        # ── ETAPA 3: Query rewriting + HyDE (M1) ──────────────────────────
        show_status("status-expandiendo", "🔄 Expandiendo consulta...")
        rewrite_data = rewrite_query(query, memory_ctx)
        query_variants = list({
            query,
            rewrite_data["rewritten"],
            rewrite_data["hyde"],
        } - {""})  # Eliminar cadenas vacías y duplicados

        # ── ETAPA 4: Retrieval híbrido RRF (M2) ───────────────────────────
        show_status("status-recuperando", "🔍 Recuperando información...")
        raw_results = hybrid_search_rrf(query, query_variants, use_feedback=False)

        # Si es retroalimentación, también buscar en feedback
        if intent == "retroalimentacion":
            fb_results = hybrid_search_rrf(query, query_variants, use_feedback=True)
            raw_results = raw_results + fb_results

        # ── ETAPA 5: Reranking (M3) ────────────────────────────────────────
        show_status("status-reranking", "📊 Rerankeando resultados...")
        reranked = rerank_results(query, raw_results)

        context = "\n\n---\n\n".join(r["text"] for r in reranked)[:4500]
        sources = list({r["source"] for r in reranked if r["source"] != "desconocido"})
        results["sources"] = sources

        # ── ETAPA 6: Routing de agentes (M5) ──────────────────────────────
        agent_type = route_query(query)
        results["agent_type"] = agent_type

        # ── ETAPA 7: Corrección si es retroalimentación ────────────────────
        if intent == "retroalimentacion" and last_answer:
            show_status("status-corrigiendo", "🔄 Corrigiendo con tu feedback...")
            corrected_answer = self.answer_agent.generate_correction(
                query, last_answer, context
            )

            # Guardar corrección en colección feedback (M9)
            save_feedback_dedup(
                query=query,
                answer=corrected_answer,
                rating=4,  # Rating por defecto para correcciones automáticas
                tags=["correccion_automatica"],
                corrected=True,
            )
            results["corrected"] = True

            # Evaluar calidad de la corrección (M7)
            show_status("status-evaluando", "🔬 Evaluando calidad...")
            scores = evaluate_response(query, context, corrected_answer)
            results["scores"] = scores
            results["metrics"] = {
                "latency":   round(time.time() - start, 2),
                "intent":    intent,
                "corrected": True,
                "agent":     agent_type,
                "chunks":    len(reranked),
            }

            # Devolver como generator de un solo elemento
            def single_token():
                yield corrected_answer

            results["tokens"] = single_token()
            return results

        # ── ETAPA 8: Generación con streaming real (M6 + M8) ─────────────
        show_status("status-generando", "✍️ Generando respuesta...")
        token_gen = self.answer_agent.stream(
            query=query,
            context=context,
            memory_ctx=memory_ctx,
            agent_type=agent_type,
            sources=sources,
        )
        results["tokens"] = token_gen
        results["context_used"] = context[:1200]

        # Métricas pre-evaluación (la evaluación se hace post-streaming)
        results["metrics"] = {
            "latency":   round(time.time() - start, 2),
            "intent":    intent,
            "corrected": False,
            "agent":     agent_type,
            "chunks":    len(reranked),
            "context_chars": len(context),
        }
        return results

rag_system = RAGSystemV3()

# ══════════════════════════════════════════════════════════════════════════════
# CONTADOR DE VISITAS
# ══════════════════════════════════════════════════════════════════════════════
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
    st.session_state.visits  = visits
else:
    st.session_state.visits = load_counter()

# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, default in [
    ("messages",        []),
    ("metrics",         {"latency": 0, "intent": "pregunta", "corrected": False, "agent": "general"}),
    ("last_scores",     {}),
    ("pending_feedback", None),  # Guarda la última respuesta para feedback
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.title("💬 Chat Académico EISC")

# Mostrar historial de mensajes
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Subir documento + métricas + feedback
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Subir Documento")
    uploaded_file = st.file_uploader(
        "PDF sobre acreditación",
        type=["pdf"],
        help="Se añadirá al índice vectorial automáticamente",
    )
    if uploaded_file:
        if st.button("🚀 Procesar e Indexar", type="primary"):
            pdf_bytes = uploaded_file.read()

            # M12.3: Verificar si el PDF ya fue indexado
            pdf_hash = _hash_pdf(pdf_bytes)
            if _check_pdf_already_indexed(pdf_hash):
                st.warning(
                    f"⚠️ Este documento ya fue indexado previamente.\n"
                    f"Hash: `{pdf_hash[:12]}...`"
                )
            else:
                with st.spinner("Extrayendo y fragmentando contenido..."):
                    chunks, doc_hash = process_uploaded_document(
                        pdf_bytes, uploaded_file.name
                    )

                if chunks:
                    st.success(
                        f"✅ {len(chunks)} chunks extraídos "
                        f"({sum(1 for c in chunks if c['type'] == 'table')} tablas, "
                        f"{sum(1 for c in chunks if c['type'] == 'text')} texto)"
                    )

                    # M12.6: Barra de progreso durante indexación
                    progress = st.progress(0, text="Preparando indexación...")
                    stats = add_chunks_to_qdrant(
                        chunks=chunks,
                        filename=uploaded_file.name,
                        pdf_hash=doc_hash,
                        progress_bar=progress,
                    )
                    progress.progress(1.0, text="Indexación completa")

                    # M12.7: Mostrar estadísticas
                    col_n, col_d, col_e = st.columns(3)
                    col_n.metric("Nuevos", stats["nuevos"])
                    col_d.metric("Duplicados", stats["duplicados"])
                    col_e.metric("Errores", stats["errores"])

                    if stats["nuevos"] > 0:
                        # Invalidar caché BM25 para reflejar nuevo contenido
                        build_bm25_index.clear()
                        st.balloons()

                    if stats["duplicados"] > 0:
                        st.info(
                            f"ℹ️ {stats['duplicados']} chunks ya existían "
                            f"y no se duplicaron."
                        )
                else:
                    st.warning("⚠️ No se extrajeron chunks válidos")

    st.markdown("---")

    # Perfil de usuario
    col1, col2 = st.columns([1, 2])
    with col1:
        avatar_path = "data/yo.webp"
        if os.path.exists(avatar_path):
            st.image(avatar_path, width=70)
        else:
            st.markdown("👤")
    with col2:
        st.markdown(f"**{st.session_state.user}**")
        st.caption("EISC · Univalle")

    st.markdown("### 📊 Métricas de sesión")
    metrics = st.session_state.metrics
    st.metric("⏱️ Latencia",     f"{metrics.get('latency', 0)} s")
    st.metric("🤖 Agente",       metrics.get('agent', 'general').capitalize())
    st.metric("🔍 Intención",    metrics.get('intent', 'pregunta').capitalize())
    st.metric("📚 Chunks usados", metrics.get('chunks', 0))
    st.metric("👥 Visitas",       st.session_state.visits)

    # Scores de calidad de la última respuesta
    if st.session_state.last_scores:
        st.markdown("### 🔬 Calidad de última respuesta")
        scores = st.session_state.last_scores
        st.progress(scores.get("faithfulness", 0.8),      text=f"Faithfulness: {scores.get('faithfulness', 0.8):.0%}")
        st.progress(scores.get("answer_relevance", 0.8),  text=f"Relevance: {scores.get('answer_relevance', 0.8):.0%}")
        st.progress(scores.get("context_precision", 0.7), text=f"Precisión ctx: {scores.get('context_precision', 0.7):.0%}")
        halluc = scores.get("hallucination_risk", 0.2)
        color = "🟢" if halluc < 0.3 else ("🟡" if halluc < 0.5 else "🔴")
        st.caption(f"{color} Riesgo de alucinación: {halluc:.0%}")

    st.markdown("---")
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_scores = {}
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# M8 · MANEJO DE INPUT CON STREAMING REAL
# ══════════════════════════════════════════════════════════════════════════════
avatar_b64 = get_base64_image("data/yo.webp") or ""
prompt = st.chat_input("Escribe tu pregunta sobre acreditación EISC...")

if prompt:
    # M11: Verificar rate limit
    if not check_rate_limit(st.session_state.user):
        st.warning(f"⚠️ Límite de {MAX_REQUESTS_PER_MINUTE} consultas/minuto alcanzado. Espera un momento.")
        st.stop()

    # M11: Sanitizar query
    prompt_clean = sanitize_query(prompt)

    # Recuperar última respuesta del asistente para contexto
    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = re.sub(r"<[^>]+>", "", m.get("content", ""))[:800]
            break

    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_clean})
    with st.chat_message("user"):
        st.markdown(prompt_clean)

    # Placeholder para el avatar de estado
    status_ph = st.empty()

    # Ejecutar pipeline RAG
    rag_result = rag_system.run_stream(
        query=prompt_clean,
        messages=st.session_state.messages,
        last_answer=last_answer,
        status_placeholder=status_ph,
    )

    # ── M8: Streaming de respuesta ──────────────────────────────────────────
    status_ph.markdown(
        f'<div class="thinking-avatar status-generando">'
        f'{"<img src=data:image/webp;base64," + avatar_b64 + " class=avatar-img>" if avatar_b64 else ""}'
        f'<span>✍️ Generando respuesta...</span></div>',
        unsafe_allow_html=True,
    )

    full_answer = ""
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()

        # Corrección: mostrar directamente sin streaming visual
        if rag_result["corrected"]:
            for token in rag_result["tokens"]:
                full_answer += token
            st.markdown(
                '<span style="color:#e65100;font-weight:bold;">✏️ Respuesta corregida según tu feedback</span>',
                unsafe_allow_html=True,
            )
            stream_placeholder.markdown(full_answer)
        else:
            # Streaming token a token con cursor parpadeante
            for token in rag_result["tokens"]:
                full_answer += token
                stream_placeholder.markdown(full_answer + "▌")
            stream_placeholder.markdown(full_answer)  # Versión final sin cursor

        # ── M7: Evaluar calidad (post-streaming) ───────────────────────────
        # Reusar contexto ya recuperado — no hacer búsqueda extra
        context_for_eval = rag_result.get("context_used", full_answer[:1200])
        status_ph.markdown(
            f'<div class="thinking-avatar status-evaluando">'
            f'<span>🔬 Evaluando calidad...</span></div>',
            unsafe_allow_html=True,
        )
        scores = evaluate_response(prompt_clean, context_for_eval, full_answer)
        st.session_state.last_scores = scores
        rag_result["metrics"]["scores"] = scores

        badge_label, badge_class = quality_badge(scores)

        # Mostrar badge de calidad
        st.markdown(
            f'<span class="quality-badge {badge_class}">🔬 {badge_label}</span>',
            unsafe_allow_html=True,
        )

        # Advertencia si el riesgo de alucinación es alto
        if scores.get("hallucination_risk", 0) > HALLUCINATION_THRESHOLD:
            st.warning(
                "⚠️ Esta respuesta podría contener información no verificada. "
                "Consulta directamente los documentos originales."
            )

        # ── Mostrar fuentes ─────────────────────────────────────────────────
        sources = rag_result.get("sources", [])
        if sources:
            st.markdown('<div class="sources-container">', unsafe_allow_html=True)
            st.markdown("### 📚 Fuentes consultadas")
            badges = " ".join(
                f'<span class="source-badge">📄 {s}</span>' for s in sources
            )
            st.markdown(badges, unsafe_allow_html=True)
            agent_type = rag_result.get("agent_type", "general")
            st.caption(
                f"Agente: {agent_type} · "
                f"Chunks: {rag_result['metrics'].get('chunks', 0)} · "
                f"Latencia: {rag_result['metrics'].get('latency', 0)}s"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── M9: Botones de valoración ───────────────────────────────────────
        st.markdown("**¿Fue útil esta respuesta?**")
        rating_cols = st.columns(5)
        rating_labels = ["😞 Muy mala", "😕 Mala", "😐 Regular", "🙂 Buena", "😄 Excelente"]
        for i, (col, label) in enumerate(zip(rating_cols, rating_labels)):
            if col.button(label.split()[0], key=f"rating_btn_{i}_{len(st.session_state.messages)}", help=label):
                result_code = save_feedback_dedup(
                    query=prompt_clean,
                    answer=full_answer,
                    rating=i + 1,
                    tags=[f"rating_{i+1}", agent_type],
                )
                st.toast(f"✅ Valoración guardada ({label.split()[1]})", icon="⭐")

    # Limpiar status
    time.sleep(0.5)
    status_ph.empty()

    # Guardar en historial incluyendo indicadores HTML
    display_answer = full_answer
    if rag_result["corrected"]:
        display_answer = '<span class="feedback-indicator">✏️ Corregido</span><br>' + display_answer
    if sources:
        src_html = " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
        display_answer += f'<br><br><div class="sources-container"><strong>📚 Fuentes:</strong> {src_html}</div>'

    st.session_state.messages.append({"role": "assistant", "content": display_answer})
    st.session_state.metrics = rag_result["metrics"]

    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Universidad del Valle · Grupo GUIA · ChatAcredita PRO v3.0<br>
    RAG + Agentes + Retroalimentación Vectorial · EISC 2025
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCROLL AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<script>
function scrollToBottom() {
    const msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
    if (msgs.length > 0) {
        msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
        return true;
    }
    const main = window.parent.document.querySelector('section.main');
    if (main) { main.scrollTop = main.scrollHeight; return true; }
    return false;
}
[200, 500, 900, 1400, 2000].forEach(d => setTimeout(scrollToBottom, d));
</script>
""", unsafe_allow_html=True)
