# ═════════════════════════════════════
# ChatTesis PRO — FINAL PRO + HEADER + FEEDBACK
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

# ═════════════════════════════════════
# CSS + HEADER + ANIMACIÓN
# ═════════════════════════════════════
st.markdown("""
<style>

header {visibility:hidden;}

.custom-header {
    position:fixed;
    top:0;
    left:0;
    right:0;
    height:70px;
    background:linear-gradient(90deg,#DC143C,#8B0000);
    display:flex;
    align-items:center;
    justify-content:center;
    z-index:9999;
    color:white;
    font-weight:600;
    font-size:18px;
}

.main { padding-top:80px; }

.logo-float {
    position:fixed;
    top:12px;
    right:20px;
    z-index:10000;
    background:white;
    border-radius:50%;
    padding:4px;
}

.footer {
    position:fixed;
    bottom:65px;
    left:0;
    right:0;
    text-align:center;
    font-size:11px;
    color:#999;
}

/* AVATAR */
.thinking-avatar {
    position: fixed;
    bottom: 90px;
    right: 20px;
    background: white;
    padding: 10px 14px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    display: flex;
    align-items: center;
    gap: 10px;
    z-index:9999;
}

.avatar-img {
    border-radius: 50%;
    width: 38px;
}

.dot {
    height: 6px;
    width: 6px;
    margin: 0 2px;
    background-color: #888;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s infinite both;
}

.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }

@keyframes blink {
    0% { opacity: .2; }
    20% { opacity: 1; }
    100% { opacity: .2; }
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="custom-header">
🎓 ChatAcredita PRO — EISC (Universidad del Valle)
</div>
""", unsafe_allow_html=True)

# LOGO
logo = get_base64_image("data/univalle_logo.png")
if logo:
    st.markdown(f"""
    <div class="logo-float">
        <img src="data:image/png;base64,{logo}" width="40">
    </div>
    """, unsafe_allow_html=True)

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

# EMBEDDINGS
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

# 🔥 HYBRID SEARCH + FEEDBACK
def hybrid_search(query):

    emb = embedder.encode([query])[0]

    docs = []

    vector = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points

    docs += [r.payload["text"] for r in vector]

    try:
        feedback = qdrant.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=emb.tolist(),
            limit=2,
            with_payload=True
        ).points

        docs += [r.payload["text"] for r in feedback]

    except:
        pass

    return list(set(docs))

# AGENTES
class MultiQueryAgent:
    def run(self, query):
        prompt = f"Genera 3 reformulaciones de: {query}"
        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return clean_json(r.choices[0].message.content).get("queries",[query])

class AnswerAgent:
    def run(self, query, context):
        prompt = f"Contexto:\n{context}\n\nPregunta: {query}"
        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}]
        )
        return r.choices[0].message.content

class RAG:
    def __init__(self):
        self.multi = MultiQueryAgent()
        self.answer = AnswerAgent()

    def run(self, query):
        docs = []
        for q in self.multi.run(query):
            docs.extend(hybrid_search(q))

        context = "\n\n".join(docs[:TOP_K])
        answer = self.answer.run(query, context)

        return answer, {"latency": round(time.time(),2)}

rag = RAG()

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {"latency":0}

avatar_base64 = get_base64_image("data/yo.webp")

# CHAT
st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    avatar = "👤" if m["role"]=="user" else "🧑‍🏫"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

prompt = st.chat_input("Escribe tu pregunta...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant", avatar="🧑‍🏫"):

        thinking = st.empty()

        thinking.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>Analizando...</span>
            <div>
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        answer, metrics = rag.run(prompt)

        thinking.markdown(f"""
        <div class="thinking-avatar" style="border-left:4px solid green;">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>Respuesta lista ✅</span>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(1)
        thinking.empty()

        st.markdown(answer)

        # 🔥 FEEDBACK VECTORIAL
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Útil"):
                st.success("Gracias por tu feedback")

        with col2:
            if st.button("👎 No útil"):

                texto_feedback = f"""
                Pregunta: {prompt}
                Respuesta incorrecta: {answer}
                """

                emb = embedder.encode([texto_feedback])[0]

                try:
                    qdrant.upsert(
                        collection_name=FEEDBACK_COLLECTION,
                        points=[{
                            "id": str(uuid.uuid4()),
                            "vector": emb.tolist(),
                            "payload": {"text": texto_feedback}
                        }]
                    )
                    st.warning("Aprendiendo de esta respuesta...")
                except:
                    st.error("Error guardando feedback")

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.metrics = metrics
    st.rerun()

# SIDEBAR
col1, col2 = st.sidebar.columns([1, 2])

with col1:
    st.image("data/yo.webp", width=80)

with col2:
    st.markdown(
        "**Raúl E. Gutiérrez de Piñerez Reyes**\n"
        "<span style='color:gray; font-size:13px;'>Profesor - PLN</span>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("### 📊 Métricas")
st.sidebar.metric("⏱️ Latencia", st.session_state.metrics["latency"])

with st.sidebar.expander("🧠 Cómo usar el chatbot", expanded=True):
    st.markdown("""
- Pregunta sobre acreditación  
- Usa contexto académico  
- Puedes pedir análisis  
""")

# FOOTER
st.markdown("""
<div class="footer">
Universidad del Valle • Grupo GUIA • ChatTesis PRO
</div>
""", unsafe_allow_html=True)