# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChatAcredita PRO v3.2 — RAG + Agentes + RRF + BGE-Reranker-v2-m3         ║
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
from collections import defaultdict
from datetime import datetime
from typing import Generator, Optional, List, Dict

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from rank_bm25 import BM25Okapi
import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Nuevo: FlagEmbedding para reranker BGE
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    st.error("❌ Instala FlagEmbedding: `pip install FlagEmbedding`")
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

COLLECTION_NAME = "acreditacion"
FEEDBACK_COLLECTION = "feedback_acreditacion"
EVAL_COLLECTION = "evaluaciones_chatacredita"

TOP_K = 40          # Más candidatos para RRF + rerank
TOP_K_FINAL = 10
HALLUCINATION_THRESHOLD = 0.4

# ─────────────────────────────────────────────
# ... (Mantengo todas las secciones de seguridad, login, utilidades, CSS, etc. iguales hasta los modelos)
# Para no hacer el mensaje eterno, asumo que copias todo lo anterior y solo reemplazas desde aquí:
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# MODELOS DE EMBEDDINGS + RERANKER
# ─────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3", device="cpu")

@st.cache_resource
def load_reranker():
    # BGE-Reranker-v2-m3 — muy bueno y eficiente
    return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=["cuda:0"] if os.environ.get("CUDA_VISIBLE_DEVICES") else ["cpu"])

embedder = load_embedder()
reranker = load_reranker()

st.sidebar.success("✅ Embeddings: BGE-M3 | Reranker: BGE-Reranker-v2-m3")

# ─────────────────────────────────────────────
# BM25 + HYBRID SEARCH CON RRF
# ─────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def build_bm25_index() -> tuple:
    # ... (mismo código que tenías, lo dejo intacto)
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

def hybrid_search_rrf(query: str, query_variants: List[str], use_feedback: bool = False, k_rrf: int = 80) -> List[Dict]:
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    rrf_scores: Dict[str, float] = {}
    id_to_payload: Dict[str, Dict] = {}

    # Búsqueda vectorial (BGE-M3)
    for q in query_variants:
        try:
            emb = embedder.encode([q], normalize_embeddings=True)[0]
            results = qdrant.query_points(
                collection_name=collection, 
                query=emb.tolist(), 
                limit=TOP_K, 
                with_payload=True
            ).points
            
            for rank, r in enumerate(results):
                pid = str(r.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k_rrf + rank + 1)
                if r.payload:
                    id_to_payload[pid] = r.payload
        except Exception:
            pass

    # BM25
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
                            "text": bm25_texts[idx], 
                            "source": bm25_sources[idx]
                        }

    # Ordenar por RRF
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    results = []
    for pid, score in sorted_ids:
        payload = id_to_payload.get(pid, {})
        if payload.get("text"):
            results.append({
                "id": pid, 
                "text": payload["text"], 
                "source": payload.get("source", "desconocido"), 
                "rrf_score": round(score, 4)
            })
    return results


def rerank_with_bge_v2_m3(query: str, candidates: List[Dict]) -> List[Dict]:
    """Reranking potente con BGE-Reranker-v2-m3"""
    if not candidates:
        return []

    pairs = [[query, cand["text"]] for cand in candidates]
    
    try:
        scores = reranker.compute_score(pairs, normalize=True)  # score entre 0 y 1
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
            # Combinar con RRF score
            cand["final_score"] = 0.6 * cand.get("rerank_score", 0) + 0.4 * cand.get("rrf_score", 0)
    except Exception as e:
        st.warning(f"Error en reranker: {e}")
        for cand in candidates:
            cand["rerank_score"] = cand.get("rrf_score", 0)
            cand["final_score"] = cand["rerank_score"]

    reranked = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)
    return reranked[:TOP_K_FINAL]


# Actualizar la función de rerank existente
def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    return rerank_with_bge_v2_m3(query, results)


# ─────────────────────────────────────────────
# Resto del código (RAGSystemV3, AnswerAgentV2, etc.)
# ─────────────────────────────────────────────
# ... (El resto del archivo se mantiene igual: AnswerAgentV2, evaluate_response, 
# save_feedback_dedup, RAGSystemV3, interfaz, etc.)

# Solo actualiza la llamada en run_stream:
# Después de hybrid_search_rrf:
raw_results = hybrid_search_rrf(...)
# ...
reranked = rerank_results(query, raw_results)   # Ahora usa BGE-Reranker-v2-m3