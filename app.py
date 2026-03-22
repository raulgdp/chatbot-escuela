# ═════════════════════════════════════
# ChatTesis PRO — MULTIAGENTE COMPLETO
# ═════════════════════════════════════

import streamlit as st
import os, time, json, unicodedata, base64, uuid
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# CONFIG
st.set_page_config(page_title="ChatAcredita PRO", layout="wide")

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

# CSS + HEADER
st.markdown("""
<style>
header {visibility:hidden;}
.custom-header {
    position:fixed; top:0; left:0; right:0;
    height:70px; background:#8B0000;
    display:flex; align-items:center; justify-content:center;
    color:white; font-weight:600;
}
.main { padding-top:80px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-header'>🎓 ChatAcredita PRO</div>", unsafe_allow_html=True)

# API
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

# 🔥 CLASSIFIER AGENT
class FeedbackClassifierAgent:
    def run(self, text):
        prompt = f"""
Clasifica:
1. pregunta
2. sugerencia
3. otro

Texto: {text}

JSON:
{{"tipo":"pregunta|sugerencia|otro"}}
"""
        r = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return clean_json(r.choices[0].message.content).get("tipo","pregunta")

classifier = FeedbackClassifierAgent()

# 🔥 RETRIEVAL
def hybrid_search(query):

    emb = embedder.encode([query])[0]

    docs = []

    docs += [
        r.payload["text"]
        for r in qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=emb.tolist(),
            limit=TOP_K,
            with_payload=True
        ).points
    ]

    try:
        docs += [
            r.payload["text"]
            for r in qdrant.query_points(
                collection_name=FEEDBACK_COLLECTION,
                query=emb.tolist(),
                limit=2,
                with_payload=True
            ).points
        ]
    except:
        pass

    return docs

# 🔥 RAG
def run_rag(query):

    start = time.time()

    docs = hybrid_search(query)
    context = "\n\n".join(docs[:TOP_K])

    prompt = f"Contexto:\n{context}\n\nPregunta: {query}"

    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":prompt}]
    )

    latency = round(time.time() - start,2)

    return r.choices[0].message.content, latency

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

    tipo = classifier.run(prompt)

    # 🔥 SUGERENCIA
    if tipo == "sugerencia":

        emb = embedder.encode([prompt])[0]

        qdrant.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": emb.tolist(),
                "payload": {"text": prompt}
            }]
        )

        st.warning("💡 Sugerencia guardada")
        st.stop()

    # 🔥 PREGUNTA → RAG
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        answer, latency = run_rag(prompt)

        st.markdown(answer)
        st.caption(f"⏱️ {latency}s")

    st.session_state.messages.append({"role":"assistant","content":answer})

    st.rerun()