# ═════════════════════════════════════
# ChatTesis PRO — FINAL + FEEDBACK VISUAL INTELIGENTE
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

# CSS
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

.avatar-img { border-radius: 50%; width: 38px; }

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

# HYBRID SEARCH
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

# CLASSIFIER
class FeedbackClassifierAgent:
    def run(self, text):
        prompt = f"""
Clasifica el texto:

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

# RAG
def run_rag(query):
    start = time.time()
    docs = hybrid_search(query)
    context = "\n\n".join(docs[:TOP_K])

    r = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role":"user","content":f"Contexto:\n{context}\n\nPregunta: {query}"}]
    )

    latency = round(time.time() - start, 2)
    return r.choices[0].message.content, latency

# SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []

avatar_base64 = get_base64_image("data/yo.webp")

# CHAT
st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Pregunta o sugerencia...")

if prompt:

    feedback_triggered = False  # 🔥 NUEVO

    tipo = classifier.run(prompt)

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

        feedback_triggered = True
        st.warning("💡 Sugerencia guardada")
        st.stop()

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):

        thinking = st.empty()

        thinking.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>Analizando...</span>
        </div>
        """, unsafe_allow_html=True)

        answer, latency = run_rag(prompt)

        mensaje_feedback = "📊 Sin retroalimentación"
        if feedback_triggered:
            mensaje_feedback = "🧠 Aprendiendo del usuario"

        thinking.markdown(f"""
        <div class="thinking-avatar" style="border-left:4px solid green;">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span><b>Respuesta lista</b> ✅<br><small>{mensaje_feedback}</small></span>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(1)
        thinking.empty()

        st.markdown(answer)
        st.caption(f"⏱️ {latency}s")

        # FEEDBACK
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Útil"):
                st.success("Gracias")

        with col2:
            if st.button("👎 No útil"):

                feedback_triggered = True

                texto = f"Pregunta: {prompt}\nRespuesta incorrecta: {answer}"
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

# FOOTER
st.markdown("""
<div class="footer">
Universidad del Valle • Grupo GUIA • ChatTesis PRO
</div>
""", unsafe_allow_html=True)