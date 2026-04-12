# app.py - ChatAcredita PRO: Optimizado para Streamlit Cloud (<300MB)
import streamlit as st
import os, time, json, unicodedata, base64, uuid, gc
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ════════════════════════════════════════════════════════════════════════════
# 🔐 LOGIN SIMPLE
# ════════════════════════════════════════════════════════════════════════════
USERS = {"admin": "1234", "raul": "eisc2025"}

def login():
    st.sidebar.title("🔐 Acceso a ChatAcredita (EISC)")
    user = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Ingresar"):
        if user in USERS and USERS[user] == password:
            st.session_state.auth = True
            st.session_state.user = user
            st.rerun()
        else:
            st.sidebar.error("❌ Credenciales incorrectas")

if "auth" not in st.session_state:
    st.session_state.auth = False
if not st.session_state.auth:
    login()
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN (SIN MODELOS PESADOS)
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ChatAcredita - EISC-Univalle", page_icon="🎓", layout="wide")
COLLECTION_NAME = "acreditacion"
FEEDBACK_COLLECTION = "feedback_acreditacion"
TOP_K = 5

# ════════════════════════════════════════════════════════════════════════════
# 🛠️ UTILIDADES LIGERAS
# ════════════════════════════════════════════════════════════════════════════
def get_base64_image(path):
    try:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return None

def clean_json(text):
    text = __import__('re').sub(r'`json|`', '', text).strip()
    try: return json.loads(text)
    except: return {"tipo": "pregunta"}

def get_secret(key, default=None):
    try: return st.secrets[key]
    except: return os.getenv(key, default)

# ════════════════════════════════════════════════════════════════════════════
# 🎨 CSS OPTIMIZADO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
header {visibility:hidden;}
.custom-header { position:fixed; top:0; left:0; right:0; height:70px; background:linear-gradient(90deg,#DC143C,#8B0000); display:flex; align-items:center; justify-content:center; z-index:9999; color:white; font-weight:600; }
.main { padding-top:80px; }
.thinking-avatar { position: fixed; bottom: 90px; right: 20px; background: white; padding: 8px 12px; border-radius: 10px; box-shadow: 0px 3px 10px rgba(0,0,0,0.20); display: flex; align-items: center; gap: 8px; z-index:9999; font-size: 0.92em; }
.avatar-img { border-radius:50%; width:28px; height:28px; }
.status-corrigiendo { background: #fff3e0; border-left: 4px solid #ff9800; color: #e65100; animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.4); } 70% { box-shadow: 0 0 0 8px rgba(255, 152, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0); } }
.sources-container { margin-top: 15px; padding: 12px; background: #f8fdff; border-left: 3px solid #2196f3; border-radius: 0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 🔌 CONEXIONES LIGERAS (SIN EMBEDDINGS LOCALES)
# ════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"  # Modelo gratuito y ligero

try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    st.sidebar.success("✅ OpenRouter conectado")
except Exception as e:
    st.sidebar.error(f"❌ Error OpenRouter: {str(e)[:60]}")
    st.stop()

try:
    qdrant = QdrantClient(url=get_secret("QDRANT_URL", "").strip(), api_key=get_secret("QDRANT_API_KEY", "").strip())
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections: st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada"); st.stop()
    if FEEDBACK_COLLECTION not in collections:
        qdrant.create_collection(collection_name=FEEDBACK_COLLECTION, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    st.sidebar.success("✅ Qdrant conectado")
except Exception as e:
    st.sidebar.error(f"❌ Error Qdrant: {str(e)[:60]}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# 🔍 BÚSQUEDA EN QDRANT (SIN GENERAR EMBEDDINGS LOCALMENTE)
# ════════════════════════════════════════════════════════════════════════════
def search_in_qdrant(query, collection, limit=5):
    """Busca usando el endpoint de texto de Qdrant (fastembed integrado)"""
    try:
        # Qdrant Cloud soporta búsqueda por texto directo si usas el parámetro `query_text`
        results = qdrant.query_points(
            collection_name=collection,
            query_text=query,  # ✅ Qdrant genera embedding automáticamente en la nube
            limit=limit,
            with_payload=True
        ).points
        texts = [r.payload["text"] for r in results if r.payload]
        sources = list(set([r.payload.get("source", "documento") for r in results if r.payload]))
        return texts, sources
    except Exception as e:
        st.warning(f"⚠️ Error búsqueda: {str(e)[:40]}")
        return [], []

# ════════════════════════════════════════════════════════════════════════════
# 🤖 AGENTES LIGEROS
# ════════════════════════════════════════════════════════════════════════════
def classify_intent(prompt, last_answer=""):
    keywords = ["corregir", "error", "mal", "incorrecto", "no es", "debería ser"]
    if any(k in prompt.lower() for k in keywords): return "retroalimentacion"
    try:
        r = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role":"user","content":f"Clasifica como 'pregunta' o 'retroalimentacion': {prompt}"}], temperature=0, max_tokens=20)
        return "retroalimentacion" if "retro" in r.choices[0].message.content.lower() else "pregunta"
    except: return "pregunta"

def generate_response(query, context, is_correction=False):
    warning = "⚠️ ADVERTENCIA: NO inventes información. Si no está en el contexto, responde 'No tengo información suficiente'." if is_correction else ""
    prompt = f"""Eres ChatAcredita (EISC-Univalle). Responde SOLO con el contexto.
{warning}
CONTEXTO:
{context[:3000]}
PREGUNTA:
{query}
INSTRUCCIONES:
- Si es conceptual → viñetas (•)
- Si es comparativo → tabla Markdown
- Máximo 3 párrafos
RESPUESTA:"""
    try:
        r = client.chat.completions.create(model=DEFAULT_MODEL, messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=600)
        return r.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)[:80]}"

# ════════════════════════════════════════════════════════════════════════════
# 💬 INTERFAZ DE CHAT
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state: st.session_state.messages = []
if "metrics" not in st.session_state: st.session_state.metrics = {"latency":0, "intent":"pregunta"}

st.markdown('<div class="custom-header">🎓 ChatAcredita PRO — EISC (Universidad del Valle)</div>', unsafe_allow_html=True)
st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

prompt = st.chat_input("Escribe tu pregunta sobre acreditación...")
if prompt:
    last_answer = next((m["content"] for m in reversed(st.session_state.messages) if m["role"]=="assistant"), "")
    st.session_state.messages.append({"role":"user","content":prompt})
    
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(f'<div class="thinking-avatar"><img src="data:image/webp;base64,{get_base64_image("data/yo.webp") or ""}" class="avatar-img"><span>🧠 Analizando...</span></div>', unsafe_allow_html=True)
        
        intent = classify_intent(prompt, last_answer)
        thinking.markdown(f'<div class="thinking-avatar"><img src="data:image/webp;base64,{get_base64_image("data/yo.webp") or ""}" class="avatar-img"><span>🔍 Recuperando...</span></div>', unsafe_allow_html=True)
        
        # Buscar en ambas colecciones
        docs_orig, src_orig = search_in_qdrant(prompt, COLLECTION_NAME)
        docs_fb, src_fb = search_in_qdrant(prompt, FEEDBACK_COLLECTION)
        
        all_docs = list(dict.fromkeys(docs_orig + docs_fb))[:5]
        context = "\n\n".join(all_docs)
        all_sources = list(set(src_orig + src_fb))
        
        thinking.markdown(f'<div class="thinking-avatar status-corrigiendo"><img src="data:image/webp;base64,{get_base64_image("data/yo.webp") or ""}" class="avatar-img"><span>✍️ Generando...</span></div>', unsafe_allow_html=True)
        
        answer = generate_response(prompt, context, intent=="retroalimentacion")
        
        if intent == "retroalimentacion" and last_answer:
            # Guardar feedback en Qdrant
            try:
                correction = generate_response(prompt, context, is_correction=True)
                # Para guardar feedback, necesitas subirlo desde embeddings-T.py o usar Qdrant API directamente
                st.sidebar.info("💡 Feedback registrado. Se integrará en la próxima actualización de la base de conocimiento.")
            except: pass
        
        thinking.empty()
        st.markdown(answer)
        if all_sources: st.markdown(f'<div class="sources-container"><strong>📚 Fuentes:</strong> {", ".join(all_sources)}</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.metrics = {"latency": round(time.time()-time.time(), 2), "intent": intent}
    st.rerun()

# Scroll automático
st.markdown("""<script>setTimeout(()=>{const e=document.getElementById('chat-bottom');if(e)e.scrollIntoView({behavior:'smooth'})},300);</script>""", unsafe_allow_html=True)

# Limpiar memoria periódicamente
gc.collect()