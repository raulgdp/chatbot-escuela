# ═════════════════════════════════════
# ChatAcredita PRO — FIX SIDEBAR + LLM
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ═════════════════════════════════════
# CONFIG
# ═════════════════════════════════════

st.set_page_config(
    page_title="ChatAcredita PRO",
    layout="wide",
    initial_sidebar_state="expanded"  # 🔥 IMPORTANTE
)

COLLECTION_NAME = "acreditacion"
TOP_K = 3

# ═════════════════════════════════════
# CSS FIX (NO BLOQUEA SIDEBAR)
# ═════════════════════════════════════

st.markdown("""
<style>
header {visibility:hidden;}

.custom-header {
    position:fixed;
    top:0;
    left:0;
    right:0;
    height:60px;
    background:linear-gradient(90deg,#DC143C,#8B0000);
    display:flex;
    align-items:center;
    justify-content:center;
    z-index:1000;
    color:white;
    font-weight:600;
}

/* 🔥 IMPORTANTE: no tapar sidebar */
[data-testid="stSidebar"] {
    z-index: 2000;
}

/* padding correcto */
.main {
    padding-top:70px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-header">
🎓 ChatAcredita PRO — Escuela de Ingeniría de Sistemas y Comptación (EISC)
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════
# LOGO
# ═════════════════════════════════════

def load_logo(path):
    try:
        with open(path,"rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None
logo1 = load_logo("data/logo2.png")

if logo:
    st.sidebar.image(f"data:image/png;base64,{logo1}", width=120)s
        

logo = load_logo("data/univalle_logo.png")

if logo:
    st.sidebar.image(f"data:image/png;base64,{logo}", width=120)

# ═════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {"latency":0,"f1_score":0}

# ═════════════════════════════════════
# SIDEBAR (AHORA SIEMPRE VISIBLE)
# ═════════════════════════════════════

st.sidebar.title("⚙️ Configuración")

model_option = st.sidebar.selectbox(
    "Modelo LLM",
    ["gpt-4o-mini", "gpt-4o", "mistralai/mistral-large"],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown("## 📊 Métricas")

c1, c2 = st.sidebar.columns(2)
c1.metric("⚡ Latencia", st.session_state.metrics["latency"])
c2.metric("🎯 F1", st.session_state.metrics["f1_score"])

# ═════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════

def normalize_text(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.lower().split())

def clean_json(text):
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
    base_url=get_secret("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
)

qdrant = QdrantClient(
    url=get_secret("QDRANT_URL"),
    api_key=get_secret("QDRANT_API_KEY")
)

# ═════════════════════════════════════
# EMBEDDING
# ═════════════════════════════════════

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

embedder = load_embedder()

# ═════════════════════════════════════
# RETRIEVER
# ═════════════════════════════════════

def vector_search(query):
    emb = embedder.encode([query])[0]
    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points
    return [r.payload["text"] for r in res]

# ═════════════════════════════════════
# F1 SCORE
# ═════════════════════════════════════

def compute_f1(answer, context):
    a = set(normalize_text(answer).split())
    c = set(normalize_text(context).split())

    if not a or not c:
        return 0

    p = len(a & c) / len(a)
    r = len(a & c) / len(c)

    if p + r == 0:
        return 0

    return round(2*(p*r)/(p+r),3)

# ═════════════════════════════════════
# LLM
# ═════════════════════════════════════

def llm(prompt, temp=0):
    r = client.chat.completions.create(
        model=model_option,
        messages=[{"role":"user","content":prompt}],
        temperature=temp
    )
    return r.choices[0].message.content

# ═════════════════════════════════════
# RAG SIMPLE
# ═════════════════════════════════════

def run_rag(query):

    start = time.time()

    docs = vector_search(query)
    context = "\n\n".join(docs)

    answer = llm(f"Responde con este contexto:\n{context}\nPregunta:{query}",0.2)

    latency = round(time.time()-start,2)
    f1 = compute_f1(answer, context)

    return answer, latency, f1

# ═════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════

left, center = st.columns([1,3])



# CHAT
with center:

    st.title("🎓 ChatAcredita PRO")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Pregunta sobre acreditación...")

    if prompt:

        st.session_state.messages.append({"role":"user","content":prompt})

        with st.chat_message("assistant"):

            with st.spinner("🤖 Procesando..."):
                answer, latency, f1 = run_rag(prompt)

            st.markdown(answer)

            st.session_state.metrics = {
                "latency":latency,
                "f1_score":f1
            }

        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()