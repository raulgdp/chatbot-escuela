# ═════════════════════════════════════
# ChatTesis PRO — MULTIAGENTE + AUTO-MEJORA
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

# CSS (MISMO)
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
.avatar-img {border-radius: 50%;width: 38px;}
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

# 🔥 CLASSIFIER CONTEXTUAL
def classify_intent(prompt, last_answer=""):
    prompt_llm = f"""
Contexto:
Respuesta previa: {last_answer}

Mensaje: {prompt}

Clasifica:
- pregunta
- retroalimentacion
- otro

JSON:
{{"tipo":"pregunta|retroalimentacion|otro"}}
"""
    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":prompt_llm}],
        temperature=0
    )
    return clean_json(r.choices[0].message.content).get("tipo","pregunta")

# 🔍 SEARCH
def hybrid_search(query):
    emb = embedder.encode([query])[0]
    docs = []

    docs += [r.payload["text"] for r in qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(), limit=TOP_K, with_payload=True).points]

    try:
        docs += [r.payload["text"] for r in qdrant.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=emb.tolist(), limit=2, with_payload=True).points]
    except:
        pass

    return docs

# 🤖 RAG
def run_rag(query):
    docs = hybrid_search(query)
    context = "\n\n".join(docs[:TOP_K])

    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":f"{context}\n\n{query}"}]
    )
    return r.choices[0].message.content

# 🔁 AUTO-MEJORA
def improve_answer(query, prev_answer):
    prompt = f"""
El usuario indicó que esta respuesta es incorrecta:

{prev_answer}

Corrige y mejora la respuesta para la pregunta:
{query}
"""
    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":prompt}]
    )
    return r.choices[0].message.content

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

avatar = get_base64_image("data/yo.webp")

# CHAT
st.title("💬 Chat Académico")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pregunta o sugerencia...")

if prompt:

    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = m["content"]
            break

    tipo = classify_intent(prompt, last_answer)

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(f"""
        <div class="thinking-avatar">
        <img src="data:image/webp;base64,{avatar}" class="avatar-img">
        <span>🧠 Analizando intención...</span>
        </div>""", unsafe_allow_html=True)

        time.sleep(0.5)

        thinking.markdown(f"""
        <div class="thinking-avatar">
        <img src="data:image/webp;base64,{avatar}" class="avatar-img">
        <span>🔍 Recuperando información...</span>
        </div>""", unsafe_allow_html=True)

        answer = run_rag(prompt)

        if tipo == "retroalimentacion":
            thinking.markdown(f"""
            <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar}" class="avatar-img">
            <span>🔁 Corrigiendo respuesta...</span>
            </div>""", unsafe_allow_html=True)

            answer = improve_answer(prompt, last_answer)

        thinking.markdown(f"""
        <div class="thinking-avatar" style="border-left:4px solid green;">
        <img src="data:image/webp;base64,{avatar}" class="avatar-img">
        <span>✅ Respuesta lista</span>
        </div>""", unsafe_allow_html=True)

        time.sleep(1)
        thinking.empty()

        st.markdown(answer)

        # SCROLL (MISMO)
        st.markdown("""
        <script>
        setTimeout(()=>window.scrollTo(0,document.body.scrollHeight),100);
        </script>
        """, unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

# FOOTER
st.markdown("<div class='footer'>Universidad del Valle • GUIA</div>", unsafe_allow_html=True)