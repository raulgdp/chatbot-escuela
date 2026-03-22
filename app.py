# ═════════════════════════════════════
# ChatTesis PRO — RAG + Feedback Vectorial
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64, uuid
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# CONFIG
st.set_page_config(page_title="ChatTesis PRO", layout="wide")

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

# CSS
st.markdown("""
<style>
.thinking-avatar {
 position: fixed;
 bottom: 90px;
 right: 20px;
 background: white;
 padding: 10px;
 border-radius: 12px;
 box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
 display: flex;
 align-items: center;
 gap: 10px;
 z-index:9999;
}
.avatar-img { border-radius:50%; width:38px; }
.dot { height:6px; width:6px; background:#888; border-radius:50%; animation: blink 1.4s infinite; }
@keyframes blink {0%{opacity:.2;}20%{opacity:1;}100%{opacity:.2;}}
</style>
""", unsafe_allow_html=True)

# API
client = OpenAI(api_key=get_secret("OPENAI_API_KEY"),
                base_url=get_secret("OPENAI_API_BASE"))

qdrant = QdrantClient(url=get_secret("QDRANT_URL"),
                      api_key=get_secret("QDRANT_API_KEY"))

# EMBEDDINGS
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3")

embedder = load_embedder()

# HYBRID SEARCH EXTENDIDO
def hybrid_search(query):

    emb = embedder.encode([query])[0]

    docs = []

    # 🔹 Docs oficiales
    vector_docs = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points

    docs += [r.payload["text"] for r in vector_docs]

    # 🔥 Feedback (peso menor)
    try:
        feedback_docs = qdrant.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=emb.tolist(),
            limit=2,
            with_payload=True
        ).points

        docs += [r.payload["text"] for r in feedback_docs]

    except:
        pass

    return docs

# AGENTES
class MultiQueryAgent:
    def run(self, query):
        prompt = f"Genera 3 reformulaciones de: {query}"
        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return [query]

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

avatar_base64 = get_base64_image("data/yo.webp")

# CHAT
st.title("💬 Chat Académico")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pregunta...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>Analizando...</span>
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
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

        # 🔥 FEEDBACK
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Útil"):
                st.success("Gracias")

        with col2:
            if st.button("👎 No útil"):

                texto = f"Pregunta: {prompt}\nRespuesta: {answer}"

                emb = embedder.encode([texto])[0]

                qdrant.upsert(
                    collection_name=FEEDBACK_COLLECTION,
                    points=[{
                        "id": str(uuid.uuid4()),
                        "vector": emb.tolist(),
                        "payload": {"text": texto}
                    }]
                )

                st.warning("Aprendiendo de este error...")

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()