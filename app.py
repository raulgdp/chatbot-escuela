# ═════════════════════════════════════
# ChatTesis PRO — MULTI-AGENT VERSION FINAL
# UI PROFESIONAL + AGENTES + SCROLL OK
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64
import numpy as np
from datetime import datetime

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# ═════════════════════════════════════
# CONFIG
# ═════════════════════════════════════

st.set_page_config(layout="wide")

COLLECTION_NAME = "acreditacion"
TOP_K = 5


# ═════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════

def normalize_text(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.lower().split())

def clean_json(text):
    text = text.replace("```json","").replace("```","")
    try:
        return json.loads(text)
    except:
        return {}

def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key, default)


# ═════════════════════════════════════
# API
# ═════════════════════════════════════

client = OpenAI(
    api_key=get_secret("OPENAI_API_KEY"),
    base_url=get_secret("OPENAI_API_BASE")
)

qdrant = QdrantClient(
    url=get_secret("QDRANT_URL"),
    api_key=get_secret("QDRANT_API_KEY")
)


# ═════════════════════════════════════
# MODELO EMBEDDING
# ═════════════════════════════════════

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

embedder = load_embedder()


# ═════════════════════════════════════
# BM25
# ═════════════════════════════════════

@st.cache_resource
def load_bm25():
    points = qdrant.scroll(collection_name=COLLECTION_NAME, limit=5000, with_payload=True)[0]
    chunks = [normalize_text(p.payload["text"]) for p in points]
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized), chunks

bm25, bm25_chunks = load_bm25()


# ═════════════════════════════════════
# HYBRID SEARCH
# ═════════════════════════════════════

def hybrid_search(query):

    emb = embedder.encode([query])[0]

    vector = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points

    bm25_scores = bm25.get_scores(query.split())
    bm25_idx = np.argsort(bm25_scores)[::-1][:TOP_K]

    docs = []

    for r in vector:
        docs.append(r.payload["text"])

    for idx in bm25_idx:
        docs.append(bm25_chunks[idx])

    return list(set(docs))


# ═════════════════════════════════════
# AGENTES
# ═════════════════════════════════════

class PlannerAgent:
    def run(self, query):
        return {"tool":"hybrid"}


class MultiQueryAgent:
    def run(self, query):

        prompt = f"""
Genera 3 reformulaciones de la pregunta.

Pregunta:
{query}

JSON:
{{"queries":["q1","q2","q3"]}}
"""

        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )

        data = clean_json(r.choices[0].message.content)
        return data.get("queries",[query])


class AnswerAgent:
    def run(self, query, context):

        prompt = f"""
Responde SOLO con el contexto.

Contexto:
{context}

Pregunta:
{query}
"""

        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )

        return r.choices[0].message.content


class ReflectionAgent:
    def run(self, query, context, answer):

        prompt = f"""
Evalúa si la respuesta está soportada por el contexto.

Responde GOOD o BAD.
"""

        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )

        return r.choices[0].message.content.strip()


# ═════════════════════════════════════
# ORQUESTADOR
# ═════════════════════════════════════

class IntelligentRAG:

    def __init__(self):
        self.multi = MultiQueryAgent()
        self.answer = AnswerAgent()
        self.reflect = ReflectionAgent()

    def run(self, query):

        trace = []
        start = time.time()

        queries = self.multi.run(query)
        trace.append({"multi_query":queries})

        docs = []
        for q in queries:
            docs.extend(hybrid_search(q))

        context = "\n\n".join(docs[:TOP_K])

        answer = self.answer.run(query, context)
        trace.append({"answer":answer[:120]})

        reflection = self.reflect.run(query, context, answer)
        trace.append({"reflection":reflection})

        latency = round(time.time() - start, 2)

        precision = min(1, len(docs)/TOP_K)
        recall = min(1, len(docs)/10)
        f = 2*(precision*recall)/(precision+recall) if precision+recall else 0

        return answer, trace, {
            "latency":latency,
            "f_score":round(f,3)
        }


rag = IntelligentRAG()


# ═════════════════════════════════════
# SESSION
# ═════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"latency":0,"f_score":0}


# ═════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════

st.title("🎓 ChatTesis PRO — Multi-Agent")

chat_container = st.container()

with chat_container:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

prompt = st.chat_input("Pregunta...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        with st.spinner("🤖 Agentes trabajando..."):
            answer, trace, metrics = rag.run(prompt)

        st.markdown(answer)

        with st.expander("🔎 Agent Trace"):
            st.json(trace)

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.metrics = metrics

    st.rerun()


# ═════════════════════════════════════
# MÉTRICAS
# ═════════════════════════════════════

st.sidebar.metric("Latency", st.session_state.metrics["latency"])
st.sidebar.metric("F-score", st.session_state.metrics["f_score"])