# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO v3.4 — RAG + Agentes + Retroalimentación Vectorial      ║
# ║  + BGE-Reranker-v2-m3 (neuronal) + Groq Llama 3.3 70B                      ║
# ║  EISC — Universidad del Valle, Cali, Colombia                             ║
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
import concurrent.futures
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Generator, Optional
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from rank_bm25 import BM25Okapi
import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# DIAGNÓSTICO GLOBAL
# ─────────────────────────────────────────────
try:
    import torch
except Exception as e:
    torch = None

# ─────────────────────────────────────────────
# CONFIGURACIÓN GROQ (PRINCIPAL)
# ─────────────────────────────────────────────
def get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
        return os.getenv(key, default)
    except Exception:
        return default

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────
# SEGURIDAD
# ─────────────────────────────────────────────
USERS_HASHED = {
    "admin": hashlib.sha256("1234".encode()).hexdigest(),
    "raul": hashlib.sha256("eisc2025".encode()).hexdigest(),
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
    page_title="ChatAcredita PRO v3.4 - EISC-Univalle",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLLECTION_NAME = "acreditacion"
FEEDBACK_COLLECTION = "feedback_acreditacion"
EVAL_COLLECTION = "evaluaciones_chatacredita"
TOP_K = 15
TOP_K_FINAL = 10
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
.quality-badge {
    display: inline-block; font-size: 0.78em;
    padding: 2px 7px; border-radius: 10px; margin-left: 6px;
}
.q-high { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.q-med { background:#fff8e1; color:#f57f17; border:1px solid #ffe082; }
.q-low { background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
🎓 ChatAcredita PRO v3.4 — EISC (Universidad del Valle) · Groq Llama 3.3 70B
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONEXIÓN APIs — Groq
# ─────────────────────────────────────────────
groq_available = False
client = None
try:
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        _ = client.models.list()
        groq_available = True
        st.sidebar.success(f"✅ Groq: {DEFAULT_MODEL}")
    else:
        st.sidebar.error("❌ OPENAI_API_KEY no configurada")
except Exception as e:
    st.sidebar.error(f"❌ Groq: {str(e)[:80]}")

# ─────────────────────────────────────────────
# CONEXIÓN Qdrant
# ─────────────────────────────────────────────
qdrant_available = False
qdrant = None
try:
    qdrant_url = get_secret("QDRANT_URL", "").strip()
    qdrant_key = get_secret("QDRANT_API_KEY", "").strip()
    
    if qdrant_url and qdrant_key:
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        existing = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in existing:
            qdrant_available = True
            st.sidebar.success(f"✅ Qdrant: {COLLECTION_NAME}")
        else:
            st.sidebar.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada")
    else:
        st.sidebar.error("❌ Qdrant credentials no configuradas")
except Exception as e:
    st.sidebar.error(f"❌ Qdrant: {str(e)[:80]}")

if not qdrant_available:
    st.stop()

# ─────────────────────────────────────────────
# MODELOS DE EMBEDDINGS + RERANKER
# ─────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3", device="cpu")

@st.cache_resource(show_spinner="🔄 Cargando BGE-Reranker-v2-m3...")
def load_reranker():
    try:
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
        return reranker
    except Exception as e:
        st.sidebar.warning(f"⚠️ BGE-Reranker no disponible: {str(e)[:80]}")
        return None

embedder = load_embedder()
reranker_model = load_reranker()

if reranker_model:
    st.sidebar.success("✅ BGE-M3 + BGE-Reranker-v2-m3")
else:
    st.sidebar.info("✅ BGE-M3 (modo heurístico)")

# ─────────────────────────────────────────────
# MEMORIA CONVERSACIONAL
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
        if not groq_available or client is None:
            return "[historial previo disponible]"
        try:
            r = client.chat.completions.create(
                model=FAST_MODEL,
                messages=[{"role": "user", "content": f"Resume en 2 oraciones este diálogo:\n{dialog}"}],
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
# QUERY REWRITING
# ─────────────────────────────────────────────
def rewrite_query(query: str, memory_ctx: str) -> dict:
    if not groq_available or client is None:
        return {"rewritten": query, "hyde": "", "keywords": [], "lang": "es"}
    
    prompt = f"""Contexto: {memory_ctx[-600:] if memory_ctx else 'Sin historial'}
Query: {query}
Responde SOLO con JSON: {{"rewritten": "", "hyde": "", "keywords": [], "lang": "es"}}"""
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=350)
        return clean_json(r.choices[0].message.content)
    except Exception:
        return {"rewritten": query, "hyde": "", "keywords": [], "lang": "es"}

# ─────────────────────────────────────────────
# ÍNDICE BM25
# ─────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def build_bm25_index():
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
        return BM25Okapi(tokenized), all_texts, all_ids, all_sources
    except Exception:
        return None, [], [], []

# ─────────────────────────────────────────────
# BÚSQUEDA HÍBRIDA CON RRF
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
        bm25, texts, ids, sources = build_bm25_index()
        if bm25:
            tokens = normalize_text(query).split()
            scores = bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOP_K]
            for rank, (idx, score) in enumerate(ranked):
                if score > 0 and idx < len(ids):
                    pid = str(ids[idx])
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
                    if pid not in id_to_payload:
                        id_to_payload[pid] = {"text": texts[idx], "source": sources[idx]}

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return [{"id": pid, "text": id_to_payload[pid]["text"], "source": id_to_payload[pid].get("source", "desconocido"), "rrf_score": round(score, 4)} for pid, score in sorted_ids if id_to_payload.get(pid, {}).get("text")]

# ─────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────
def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    
    if reranker_model is not None:
        try:
            pairs = [[query, r["text"]] for r in results]
            scores = reranker_model.predict(pairs, show_progress_bar=False)
            for i, r in enumerate(results):
                r["rerank_score"] = float(scores[i])
            return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K_FINAL]
        except Exception:
            pass
    
    # Fallback heurístico
    for r in results:
        base = r.get("rrf_score", 0.0)
        if "| " in r["text"] and r["text"].count("| ") > 6:
            base += 0.003
        if "feedback" in r.get("source", "").lower():
            base += 0.005
        q_words = set(normalize_text(query).split())
        overlap = len(q_words & set(r["text"].lower().split()))
        if overlap > 3:
            base += 0.001 * overlap
        r["rerank_score"] = base
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:TOP_K_FINAL]

# ─────────────────────────────────────────────
# AGENTES
# ─────────────────────────────────────────────
AGENT_PROMPTS = {
    "estadistica": "Presenta datos numéricos con precisión. Usa tablas cuando sea necesario.",
    "normativa": "Cita artículos o resoluciones exactas si están en el contexto.",
    "proceso": "Usa lista numerada de pasos clara y accionable.",
    "comparacion": "Usa tabla Markdown para comparar elementos.",
    "sintesis": "Sintetiza los puntos más importantes en máximo 4 viñetas.",
    "general": "Responde en prosa clara. Máximo 3 párrafos.",
}

def route_query(query: str) -> str:
    if not groq_available or client is None:
        return "general"
    prompt = f"""Clasifica: estadistica, normativa, proceso, comparacion, sintesis, general.
Pregunta: {query}
Responde solo con una palabra."""
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=20)
        agent = r.choices[0].message.content.strip().lower()
        return agent if agent in AGENT_PROMPTS else "general"
    except Exception:
        return "general"

def classify_intent(prompt: str, last_answer: str) -> str:
    if not last_answer:
        return "pregunta"
    if not groq_available or client is None:
        return "pregunta"
    p = f"""Respuesta anterior: {last_answer[:300]}
Nuevo mensaje: {prompt}
Clasifica: "pregunta" o "retroalimentacion". Solo JSON: {{"tipo": ""}}"""
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": p}], temperature=0, max_tokens=50)
        data = clean_json(r.choices[0].message.content)
        return data.get("tipo", "pregunta")
    except Exception:
        return "pregunta"

# ─────────────────────────────────────────────
# ANSWER AGENT
# ─────────────────────────────────────────────
class AnswerAgent:
    def stream(self, query: str, context: str, memory_ctx: str, agent_type: str, sources: list[str]) -> Generator[str, None, None]:
        format_instr = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["general"])
        source_list = ", ".join(set(sources)) if sources else "documentos"
        system = f"""Eres ChatAcredita, asistente de acreditación EISC, Univalle.
Reglas: Responde SOLO con información del contexto. Si no está, di: "No encontré información sobre esto."
Al usar un dato, añade [Fuente: {source_list}]. NUNCA inventes datos.
Formato: {format_instr}"""
        user = f"""Historial: {memory_ctx if memory_ctx else "Sin historial"}
Contexto: {context}
Pregunta: {query}"""
        
        if not groq_available or client is None:
            yield "❌ Groq no disponible. Verifica API key."
            return
        
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2, max_tokens=1000, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"⚠️ Error: {str(e)[:100]}"
    
    def generate_correction(self, query: str, last_answer: str, context: str) -> str:
        if not groq_available or client is None:
            return "Groq no disponible para corrección."
        prompt = f"""Respuesta previa: {last_answer[:500]}
Corrección del usuario: {query}
Contexto: {context}
Genera respuesta corregida:"""
        try:
            r = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.15, max_tokens=900)
            return r.choices[0].message.content
        except Exception as e:
            return f"Error en corrección: {str(e)[:100]}"

# ─────────────────────────────────────────────
# EVALUADOR
# ─────────────────────────────────────────────
def evaluate_response(query: str, context: str, answer: str) -> dict:
    default = {"faithfulness": 0.8, "answer_relevance": 0.8, "context_precision": 0.7, "hallucination_risk": 0.2}
    if not groq_available or client is None:
        return default
    prompt = f"""Evalúa: Query: {query[:300]} | Contexto: {context[:800]} | Respuesta: {answer[:500]}
Responde SOLO JSON: {{"faithfulness": 0.0-1.0, "answer_relevance": 0.0-1.0, "context_precision": 0.0-1.0, "hallucination_risk": 0.0-1.0}}"""
    try:
        r = client.chat.completions.create(model=FAST_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=150)
        return clean_json(r.choices[0].message.content)
    except Exception:
        return default

def quality_badge(scores: dict) -> tuple[str, str]:
    faith = scores.get("faithfulness", 0.8)
    halluc = scores.get("hallucination_risk", 0.2)
    if faith >= 0.8 and halluc <= 0.2:
        return "Alta confianza", "q-high"
    elif faith >= 0.6 and halluc <= 0.4:
        return "Confianza media", "q-med"
    return "Verificar respuesta", "q-low"

# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────
def save_feedback(query: str, answer: str, rating: int, tags: list[str]):
    try:
        combined = f"PREGUNTA: {query}\nRESPUESTA: {answer[:500]}"
        emb = embedder.encode([combined], normalize_embeddings=True)[0]
        qdrant.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()), vector=emb.tolist(),
                payload={"text": combined, "query": query, "answer": answer[:600], "rating": rating, "tags": tags, "timestamp": time.time()}
            )]
        )
    except Exception:
        pass

# ─────────────────────────────────────────────
# RAG SYSTEM
# ─────────────────────────────────────────────
class RAGSystem:
    def __init__(self):
        self.answer_agent = AnswerAgent()

    def run_stream(self, query: str, messages: list, last_answer: str = "", status_placeholder=None) -> dict:
        start = time.time()
        results = {"sources": [], "metrics": {}, "scores": {}, "intent": "pregunta", "corrected": False, "agent_type": "general", "tokens": None}

        def show_status(css_class: str, text: str):
            if status_placeholder:
                status_placeholder.markdown(f'<div class="thinking-avatar {css_class}"><span>{text}</span></div>', unsafe_allow_html=True)

        show_status("status-analizando", "🧠 Analizando intención...")
        intent = classify_intent(query, last_answer)
        results["intent"] = intent
        memory_ctx = memory.get_context(messages)
        
        show_status("status-expandiendo", "🔄 Expandiendo consulta...")
        rewrite_data = rewrite_query(query, memory_ctx)
        query_variants = list({query, rewrite_data.get("rewritten", ""), rewrite_data.get("hyde", "")} - {""})
        
        show_status("status-recuperando", "🔍 Recuperando información...")
        raw_results = hybrid_search_rrf(query, query_variants, use_feedback=False)
        
        reranker_name = "BGE-Reranker" if reranker_model else "heurísticas"
        show_status("status-reranking", f"📊 Reranking con {reranker_name}...")
        reranked = rerank_results(query, raw_results)
        context = "\n\n---\n\n".join(r["text"] for r in reranked)[:4500]
        sources = list({r["source"] for r in reranked if r["source"] != "desconocido"})
        results["sources"] = sources
        agent_type = route_query(query)
        results["agent_type"] = agent_type
        
        if intent == "retroalimentacion" and last_answer:
            show_status("status-corrigiendo", "🔄 Corrigiendo con tu feedback...")
            corrected_answer = self.answer_agent.generate_correction(query, last_answer, context)
            save_feedback(query, corrected_answer, 4, ["correccion"])
            results["corrected"] = True
            show_status("status-evaluando", "🔬 Evaluando calidad...")
            scores = evaluate_response(query, context, corrected_answer)
            results["scores"] = scores
            results["metrics"] = {"latency": round(time.time() - start, 2), "intent": intent, "corrected": True, "agent": agent_type, "chunks": len(reranked)}
            def gen(): yield corrected_answer
            results["tokens"] = gen()
            return results
        
        show_status("status-generando", "✍️ Generando respuesta con Groq Llama 3.3...")
        results["tokens"] = self.answer_agent.stream(query, context, memory_ctx, agent_type, sources)
        results["metrics"] = {"latency": round(time.time() - start, 2), "intent": intent, "corrected": False, "agent": agent_type, "chunks": len(reranked)}
        return results

rag_system = RAGSystem()

# ─────────────────────────────────────────────
# INICIALIZACIÓN SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("metrics", {"latency": 0, "intent": "pregunta", "corrected": False, "agent": "general"}),
    ("last_scores", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────────
st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 Modelo LLM")
    if groq_available:
        st.success(f"✅ Groq — {DEFAULT_MODEL}")
    else:
        st.error("❌ Groq no disponible")
    
    st.markdown("---")
    st.markdown("### 📊 Métricas de sesión")
    metrics = st.session_state.metrics
    st.metric("⏱️ Latencia", f"{metrics.get('latency', 0)} s")
    st.metric("🎯 Agente", metrics.get('agent', 'general').capitalize())
    st.metric("📚 Chunks", metrics.get('chunks', 0))
    
    if st.session_state.last_scores:
        st.markdown("### 🔬 Calidad última respuesta")
        scores = st.session_state.last_scores
        st.progress(scores.get("faithfulness", 0.8), text=f"Faithfulness: {scores.get('faithfulness', 0.8):.0%}")
        st.progress(scores.get("answer_relevance", 0.8), text=f"Relevance: {scores.get('answer_relevance', 0.8):.0%}")
        halluc = scores.get("hallucination_risk", 0.2)
        color = "🟢" if halluc < 0.3 else ("🟡" if halluc < 0.5 else "🔴")
        st.caption(f"{color} Riesgo alucinación: {halluc:.0%}")
    
    st.markdown("---")
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_scores = {}
        st.rerun()

# ─────────────────────────────────────────────
# INPUT Y RESPUESTA
# ─────────────────────────────────────────────
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
    
    status_ph.markdown('<div class="thinking-avatar status-generando"><span>✍️ Generando respuesta...</span></div>', unsafe_allow_html=True)
    
    full_answer = ""
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        if rag_result.get("corrected", False):
            for token in rag_result["tokens"]:
                full_answer += token
            st.markdown('<span style="color:#e65100;font-weight:bold;">✏️ Respuesta corregida según tu feedback</span>', unsafe_allow_html=True)
            stream_placeholder.markdown(full_answer)
        else:
            for token in rag_result["tokens"]:
                full_answer += token
                stream_placeholder.markdown(full_answer + "▌")
            stream_placeholder.markdown(full_answer)
        
        context_for_eval = full_answer[:1200]
        scores = evaluate_response(prompt_clean, context_for_eval, full_answer)
        st.session_state.last_scores = scores
        
        badge_label, badge_class = quality_badge(scores)
        st.markdown(f'<span class="quality-badge {badge_class}">🔬 {badge_label}</span>', unsafe_allow_html=True)
        
        if scores.get("hallucination_risk", 0) > HALLUCINATION_THRESHOLD:
            st.warning("⚠️ Esta respuesta podría contener información no verificada.")
        
        sources = rag_result.get("sources", [])
        if sources:
            st.markdown('<div class="sources-container">', unsafe_allow_html=True)
            st.markdown("### 📚 Fuentes consultadas")
            badges = " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
            st.markdown(badges, unsafe_allow_html=True)
            st.caption(f"Agente: {rag_result.get('agent_type', 'general')} · Chunks: {rag_result['metrics'].get('chunks', 0)} · Lat