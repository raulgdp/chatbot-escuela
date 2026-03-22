# ═════════════════════════════════════
# ChatTesis PRO — MULTIAGENTE VISIBLE + AUTO-MEJORA
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64, uuid
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# CONFIG
st.set_page_config(
    page_title="ChatAcredita - EISC-Univalle (Cali-Colombia)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLLECTION_NAME = "acreditacion"
FEEDBACK_COLLECTION = "feedback_acreditacion"
TOP_K = 5

# UTILIDADES
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

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

# 🔥 CLASSIFIER NUEVO (NO rompe nada)
def classify_feedback(prompt, last_answer=""):

    prompt_llm = f"""
Contexto:
Respuesta previa: {last_answer}

Mensaje del usuario:
{prompt}

Clasifica:
- pregunta
- retroalimentacion

JSON:
{{"tipo":"pregunta|retroalimentacion"}}
"""

    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":prompt_llm}],
        temperature=0
    )

    return clean_json(r.choices[0].message.content).get("tipo","pregunta")

# CSS (igual)
st.markdown("""<style>
header {visibility:hidden;}
.custom-header {position:fixed;top:0;left:0;right:0;height:70px;
background:linear-gradient(90deg,#DC143C,#8B0000);
display:flex;align-items:center;justify-content:center;
z-index:9999;color:white;font-weight:600;font-size:18px;}
.main { padding-top:80px; }
.footer {position:fixed;bottom:65px;left:0;right:0;text-align:center;font-size:11px;color:#999;}
.thinking-avatar {position: fixed;bottom: 90px;right: 20px;background: white;padding: 10px 14px;
border-radius: 12px;box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
display: flex;align-items: center;gap: 10px;z-index:9999;}
.avatar-img { border-radius: 50%; width: 38px; }
</style>""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='custom-header'>🎓 ChatAcredita PRO — EISC</div>", unsafe_allow_html=True)

# API
client = OpenAI(
    api_key=get_secret("OPENAI_API_KEY"),
    base_url=get_secret("OPENAI_API_BASE")
)

qdrant = QdrantClient(
    url=get_secret("QDRANT_URL"),
    api_key=get_secret("QDRANT_API_KEY")
)

# EMBEDDING
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

embedder = load_embedder()

# BM25
@st.cache_resource
def load_bm25():
    points = qdrant.scroll(collection_name=COLLECTION_NAME, limit=5000, with_payload=True)[0]
    chunks = [normalize_text(p.payload["text"]) for p in points]
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized), chunks

bm25, bm25_chunks = load_bm25()

# SEARCH
def hybrid_search(query):
    emb = embedder.encode([query])[0]
    docs = []

    docs += [r.payload["text"] for r in qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(), limit=TOP_K, with_payload=True).points]

    return docs

# RAG
def run_rag(query):
    docs = hybrid_search(query)
    context = "\n\n".join(docs[:TOP_K])

    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":f"{context}\n\n{query}"}]
    )

    return r.choices[0].message.content, {"latency":0}

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

avatar_base64 = get_base64_image("data/yo.webp")

# CHAT
st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Escribe tu pregunta...")

if prompt:

    # obtener última respuesta
    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = m["content"]
            break

    tipo = classify_feedback(prompt, last_answer)

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        thinking = st.empty()

        # 🧠
        thinking.markdown(f"""
        <div class="thinking-avatar">
        <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
        <span>🧠 Analizando</span>
        </div>""", unsafe_allow_html=True)

        time.sleep(0.3)

        # 🔍
        thinking.markdown(f"""
        <div class="thinking-avatar">
        <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
        <span>🔍 Recuperando información</span>
        </div>""", unsafe_allow_html=True)

        answer, metrics = run_rag(prompt)

        # 🔁 feedback
        if tipo == "retroalimentacion":

            thinking.markdown(f"""
            <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🔁 Corrigiendo respuesta</span>
            </div>""", unsafe_allow_html=True)

            r = client.chat.completions.create(
                model="mistralai/mistral-large",
                messages=[{"role":"user","content":f"Corrige: {answer} basado en: {prompt}"}]
            )

            answer = r.choices[0].message.content

        else:
            thinking.markdown(f"""
            <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>✍️ Generando respuesta</span>
            </div>""", unsafe_allow_html=True)

        # FINAL
        thinking.markdown(f"""
        <div class="thinking-avatar" style="border-left:4px solid green;">
        <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
        <span><b>Respuesta lista</b> ✅</span>
        </div>""", unsafe_allow_html=True)

        time.sleep(1)
        thinking.empty()

        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()