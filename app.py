# ═════════════════════════════════════
# ChatAcredita PRO — MULTI-AGENT FINAL
# SCROLL REAL + UI FIX
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

st.set_page_config(layout="wide")

COLLECTION_NAME = "acreditacion"
TOP_K = 5

# ═════════════════════════════════════
# CSS SCROLL REAL
# ═════════════════════════════════════

st.markdown("""
<style>

.chat-container {
    height: calc(100vh - 180px);
    overflow-y: auto;
    padding: 15px;
    border-radius: 10px;
    background: #fafafa;
}

section[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 10px 20%;
    border-top: 2px solid #DC143C;
    z-index: 9999;
}

</style>
""", unsafe_allow_html=True)

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
# RETRIEVAL
# ═════════════════════════════════════

def search(query):
    emb = embedder.encode([query])[0]
    res = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points
    return [r.payload["text"] for r in res]

# ═════════════════════════════════════
# AGENTES
# ═════════════════════════════════════

class PlannerAgent:
    def run(self, query):
        return {"tool":"hybrid"}

class MultiQueryAgent:
    def run(self, query):
        return [query, query+" detalle", query+" explicación"]

class AnswerAgent:
    def run(self, query, context):
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Contexto:\n{context}\nPregunta:{query}"}],
            temperature=0.2
        )
        return r.choices[0].message.content

class ReflectionAgent:
    def run(self, query, context, answer):
        return "GOOD"

# ═════════════════════════════════════
# ORQUESTADOR
# ═════════════════════════════════════

class RAG:

    def __init__(self):
        self.multi = MultiQueryAgent()
        self.answer = AnswerAgent()

    def run(self, query):

        start = time.time()

        queries = self.multi.run(query)

        docs = []
        for q in queries:
            docs.extend(search(q))

        context = "\n\n".join(docs[:TOP_K])

        answer = self.answer.run(query, context)

        latency = round(time.time() - start, 2)

        return answer, latency

rag = RAG()

# ═════════════════════════════════════
# SESSION
# ═════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

# ═════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════

st.title("🎓 ChatAcredita PRO")

# CHAT BOX
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown('</div>', unsafe_allow_html=True)

# INPUT
prompt = st.chat_input("Pregunta...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    answer, latency = rag.run(prompt)

    st.session_state.messages.append({"role":"assistant","content":answer})

    st.rerun()

# ═════════════════════════════════════
# AUTO SCROLL REAL
# ═════════════════════════════════════

st.markdown("""
<script>
setTimeout(function() {
    const chat = window.parent.document.querySelector('.chat-container');
    if(chat){
        chat.scrollTop = chat.scrollHeight;
    }
}, 200);
</script>
""", unsafe_allow_html=True)