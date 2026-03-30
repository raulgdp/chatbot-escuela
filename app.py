# app.py - ChatAcredita PRO: Avatar pequeño + Señal de corrección visible
import streamlit as st
import os
import time
import json
import unicodedata
import base64
import uuid
import re
import tempfile
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from rank_bm25 import BM25Okapi
import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ════════════════════════════════════════════════════════════════════════════
# 🔐 LOGIN SIMPLE
# ════════════════════════════════════════════════════════════════════════════
USERS = {
    "admin": "1234",
    "raul": "eisc2025"
}

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
# ⚙️ CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatAcredita - EISC-Univalle (Cali-Colombia)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLLECTION_NAME = "acreditacion"
FEEDBACK_COLLECTION = "feedback_acreditacion"
TOP_K = 5

# ════════════════════════════════════════════════════════════════════════════
# 🛠️ UTILIDADES
# ════════════════════════════════════════════════════════════════════════════
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
    text = re.sub(r'```json|```', '', text).strip()
    try:
        return json.loads(text)
    except:
        return {"tipo": "pregunta"}

def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key, default)

# ════════════════════════════════════════════════════════════════════════════
# 🎨 CSS PERSONALIZADO (AVATAR MÁS PEQUEÑO + SEÑALES DE ESTADO)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
header {visibility:hidden;}
.custom-header {
    position:fixed; top:0; left:0; right:0;
    height:70px;
    background:linear-gradient(90deg,#DC143C,#8B0000);
    display:flex; align-items:center; justify-content:center;
    z-index:9999; color:white; font-weight:600;
}
.main { padding-top:80px; }
.footer {
    position:fixed; bottom:65px; left:0; right:0;
    text-align:center; font-size:11px; color:#999;
}
.thinking-avatar {
    position: fixed; bottom: 90px; right: 20px;
    background: white; padding: 8px 12px;
    border-radius: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.20);
    display: flex; align-items: center; gap: 8px;
    z-index:9999;
    font-size: 0.92em;
}
.avatar-img {
    border-radius:50%;
    width:28px;  /* ✅ REDUCIDO DE 38px A 28px */
    height:28px;
}
.status-analizando { background: #e3f2fd; border-left: 3px solid #2196f3; color: #1565c0; }
.status-recuperando { background: #e8f5e9; border-left: 3px solid #4caf50; color: #2e7d32; }
.status-corrigiendo {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    color: #e65100;
    animation: pulse 1.5s infinite;
}
.status-generando { background: #f3e5f5; border-left: 3px solid #9c27b0; color: #4a148c; }
.status-listo {
    background: #e8f5e9;
    border-left: 4px solid #4caf50;
    color: #2e7d32;
    box-shadow: 0 0 12px rgba(76, 175, 80, 0.4);
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(255, 152, 0, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0); }
}
.source-badge {
    display: inline-block;
    background: #e3f2fd;
    color: #1976d2;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    margin: 2px;
    border: 1px solid #bbdefb;
}
.sources-container {
    margin-top: 15px;
    padding: 12px;
    background: #f8fdff;
    border-left: 3px solid #2196f3;
    border-radius: 0 8px 8px 0;
}
.feedback-indicator {
    display: inline-block;
    background: #fff3e0;
    color: #e65100;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    margin-left: 8px;
    border: 1px solid #ffcc80;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 🌐 HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="custom-header">
    🎓 ChatAcredita PRO — EISC (Universidad del Valle)
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 🔌 CONFIGURACIÓN DE APIs (CORREGIDO - MODELO VÁLIDO)
# ════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = "https://openrouter.ai/api/v1"  # ✅ Hardcoded sin espacios

# ✅ MODELO VÁLIDO Y DISPONIBLE EN OPENROUTER
DEFAULT_MODEL = "mistralai/mistral-large"  # ✅ Gratuito y estable

try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    _ = client.models.list()
    st.sidebar.success(f"✅ OpenRouter conectado ({DEFAULT_MODEL})")
except Exception as e:
    st.sidebar.error(f"❌ Error OpenRouter: {str(e)[:80]}")
    st.sidebar.info("""
    🔑 Verifica en Settings → Secrets:
    • OPENAI_API_KEY = sk-or-v1-...
    • OPENAI_API_BASE = https://openrouter.ai/api/v1 (SIN ESPACIOS)
    """)

try:
    qdrant = QdrantClient(
        url=get_secret("QDRANT_URL", "").strip(),
        api_key=get_secret("QDRANT_API_KEY", "").strip()
    )
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada")
        st.stop()
    
    if FEEDBACK_COLLECTION not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=FEEDBACK_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    st.sidebar.success(f"✅ Qdrant: {COLLECTION_NAME} + {FEEDBACK_COLLECTION}")
except Exception as e:
    st.sidebar.error(f"❌ Error Qdrant: {str(e)[:80]}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# 📦 EMBEDDING MODEL (BGE-M3 1024d)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-m3", device="cpu")

embedder = load_embedder()
st.sidebar.success("✅ Embeddings: BGE-M3 (1024d)")

# ════════════════════════════════════════════════════════════════════════════
# 📤 PROCESAMIENTO DE DOCUMENTOS SUBIDOS (SIN TESSERACT)
# ════════════════════════════════════════════════════════════════════════════
def process_uploaded_document(pdf_bytes, filename):
    """Procesa PDF SIN Tesseract (solo PyMuPDF4LLM)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        doc = fitz.open(tmp_path)
        all_text = pymupdf4llm.to_markdown(doc)
        doc.close()
        os.unlink(tmp_path)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n## ", "\n\n### ", "\n\n|", "\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )
        
        chunks = splitter.split_text(all_text)
        valid_chunks = [c.strip() for c in chunks if len(c.strip()) > 100]
        
        return valid_chunks, [filename] * len(valid_chunks)
    except Exception as e:
        st.error(f"❌ Error procesando documento: {str(e)[:100]}")
        return [], []

def add_chunks_to_qdrant(chunks, sources):
    """Añade chunks a Qdrant Cloud"""
    try:
        chunks_normalized = [normalize_text(chunk) for chunk in chunks]
        embeddings = embedder.encode(chunks_normalized, normalize_embeddings=True)
        
        points = []
        for i, (chunk, source, embedding) in enumerate(zip(chunks_normalized, sources, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk_id": i,
                        "type": "documento_subido",
                        "timestamp": time.time()
                    }
                )
            )
        
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"✅ Añadidos {len(points)} chunks a Qdrant Cloud")
        return True
    except Exception as e:
        st.error(f"❌ Error subiendo a Qdrant: {str(e)[:100]}")
        return False

# ════════════════════════════════════════════════════════════════════════════
# 🔍 BÚSQUEDA HÍBRIDA CON FUENTES
# ════════════════════════════════════════════════════════════════════════════
def hybrid_search_with_sources(query, use_feedback=False):
    """Busca en Qdrant y devuelve texto + fuentes"""
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    
    try:
        emb = embedder.encode([query], normalize_embeddings=True)[0]
        results = qdrant.query_points(
            collection_name=collection,
            query=emb.tolist(),
            limit=TOP_K,
            with_payload=True
        ).points
        
        texts = []
        sources = set()
        
        for r in results:
            if r.payload:
                texts.append(r.payload["text"])
                source = r.payload.get("source", "Documento desconocido")
                sources.add(source)
        
        return texts, list(sources)
    except Exception as e:
        st.warning(f"⚠️ Error búsqueda: {str(e)[:50]}")
        return [], []

# ════════════════════════════════════════════════════════════════════════════
# 🤖 AGENTES
# ════════════════════════════════════════════════════════════════════════════
def classify_feedback(prompt, last_answer=""):
    prompt_llm = f"""
Contexto:
Respuesta previa: {last_answer[:300]}

Nuevo mensaje del usuario:
{prompt}

Clasifica:
- Si el usuario hace una nueva pregunta → "pregunta"
- Si el usuario corrige o mejora la respuesta anterior → "retroalimentacion"

JSON: {{"tipo": "pregunta" o "retroalimentacion"}}
"""
    try:
        r = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt_llm}],
            temperature=0,
            max_tokens=50
        )
        result = clean_json(r.choices[0].message.content)
        return result.get("tipo", "pregunta")
    except:
        return "pregunta"

class AnswerAgent:
    def run(self, query, context):
        prompt = f"""
Eres ChatAcredita, asistente especializado en acreditación de la EISC.
Responde SOLO con la información del contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA:
{query}

INSTRUCCIONES:
- Si es conceptual → viñetas (•)
- Si es comparativo → tabla Markdown
- Máximo 3 párrafos
- NO menciones las fuentes en tu respuesta (se mostrarán después automáticamente)
"""
        try:
            r = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"⚠️ Error: {str(e)[:100]}"

class RAGSystem:
    def __init__(self):
        self.answer_agent = AnswerAgent()
    
    def run(self, query, last_answer=""):
        start = time.time()
        intent = classify_feedback(query, last_answer)
        
        # Buscar en colecciones relevantes
        if intent == "retroalimentacion":
            docs_original, sources_original = hybrid_search_with_sources(query, use_feedback=False)
            docs_feedback, sources_feedback = hybrid_search_with_sources(query, use_feedback=True)
            context = "\n\n".join(docs_original + docs_feedback)[:4000]
            all_sources = list(set(sources_original + sources_feedback))
        else:
            docs, all_sources = hybrid_search_with_sources(query, use_feedback=False)
            context = "\n\n".join(docs)[:4000]
        
        answer = self.answer_agent.run(query, context)
        latency = round(time.time() - start, 2)
        
        # Corregir si es retroalimentación
        if intent == "retroalimentacion" and last_answer:
            try:
                r = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role": "user", "content": f"Corrige: {last_answer} basado en: {query}"}],
                    temperature=0.2,
                    max_tokens=800
                )
                corrected = r.choices[0].message.content
                
                # Guardar en feedback collection
                emb = embedder.encode([corrected], normalize_embeddings=True)[0]
                qdrant.upsert(
                    collection_name=FEEDBACK_COLLECTION,
                    points=[PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload={
                            "text": f"PREGUNTA: {query}\n\nRESPUESTA CORREGIDA: {corrected}\n\nRETROALIMENTACIÓN: {query}",
                            "source": "feedback_usuario",
                            "type": "respuesta_corregida",
                            "timestamp": time.time()
                        }
                    )]
                )
                return corrected, all_sources, {"latency": latency, "intent": intent, "corrected": True}
            except:
                pass
        
        return answer, all_sources, {"latency": latency, "intent": intent, "corrected": False}

rag_system = RAGSystem()

# ════════════════════════════════════════════════════════════════════════════
# 👥 CONTADOR DE VISITAS
# ════════════════════════════════════════════════════════════════════════════
COUNTER_FILE = "counter.json"
def load_counter():
    try:
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)["visits"]
    except:
        return 0

def save_counter(value):
    with open(COUNTER_FILE, "w") as f:
        json.dump({"visits": value}, f)

if "counted" not in st.session_state:
    visits = load_counter() + 1
    save_counter(visits)
    st.session_state.counted = True
    st.session_state.visits = visits
else:
    st.session_state.visits = load_counter()

# ════════════════════════════════════════════════════════════════════════════
# 💬 INTERFAZ DE CHAT
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"latency": 0, "intent": "pregunta", "corrected": False}

st.title("💬 Chat Académico EISC")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 📁 SIDEBAR: Subir documento
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Subir Documento")
    
    uploaded_file = st.file_uploader(
        "Sube PDF sobre acreditación",
        type=["pdf"],
        help="El documento será procesado y añadido a la base de conocimiento"
    )
    
    if uploaded_file:
        if st.button("🚀 Procesar y Añadir a Qdrant", type="primary"):
            with st.spinner("Procesando documento..."):
                pdf_bytes = uploaded_file.read()
                chunks, sources = process_uploaded_document(pdf_bytes, uploaded_file.name)
                
                if chunks:
                    st.success(f"✅ Extraídos {len(chunks)} chunks")
                    if add_chunks_to_qdrant(chunks, sources):
                        st.balloons()
                        st.rerun()
                else:
                    st.warning("⚠️ No se extrajeron chunks del documento")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("data/yo.webp" if os.path.exists("data/yo.webp") else "https://via.placeholder.com/80", width=80)
    with col2:
        st.markdown(f"**{st.session_state.user}**")
        st.caption("EISC Univalle")
    
    st.markdown("### 📊 Métricas")
    st.metric("⏱️ Latencia", f"{st.session_state.metrics.get('latency', 0)} s")
    st.metric("🔍 Última intención", st.session_state.metrics.get('intent', 'pregunta').capitalize())
    st.metric("👥 Visitas", st.session_state.visits)

# ════════════════════════════════════════════════════════════════════════════
# 📥 MANEJO DE INPUT DEL USUARIO (CON AVATAR PEQUEÑO + SEÑAL DE CORRECCIÓN)
# ════════════════════════════════════════════════════════════════════════════
avatar_base64 = get_base64_image("data/yo.webp") or "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhESMIAAAAABJRU5ErkJggg=="

prompt = st.chat_input("Escribe tu pregunta sobre acreditación...")

if prompt:
    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = m["content"]
            break
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    thinking = st.empty()
    
    # ✅ ETAPA 1: Analizando intención (avatar pequeño)
    with thinking.container():
        st.markdown(f"""
        <div class="thinking-avatar status-analizando">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🧠 Analizando intención...</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    # Clasificar intención
    intent = classify_feedback(prompt, last_answer)
    
    # ✅ ETAPA 2: Recuperando información
    with thinking.container():
        st.markdown(f"""
        <div class="thinking-avatar status-recuperando">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🔍 Recuperando información...</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    # Generar respuesta + fuentes
    answer, sources, metrics = rag_system.run(prompt, last_answer)
    
    # ✅ ETAPA 3: Corrigiendo respuesta (SEÑAL CLARA DE CORRECCIÓN)
    if metrics.get("corrected", False):
        with thinking.container():
            st.markdown(f"""
            <div class="thinking-avatar status-corrigiendo">
                <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
                <span>🔄 Corrigiendo respuesta con tu feedback...</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.0)
    else:
        with thinking.container():
            st.markdown(f"""
            <div class="thinking-avatar status-generando">
                <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
                <span>✍️ Generando respuesta...</span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
    
    # ✅ ETAPA 4: Respuesta lista
    with thinking.container():
        st.markdown(f"""
        <div class="thinking-avatar status-listo">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>✅ Respuesta lista</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.8)
    
    thinking.empty()
    
    # Mostrar respuesta con fuentes y señal de corrección si aplica
    with st.chat_message("assistant"):
        if metrics.get("corrected", False):
            st.markdown(f'<span style="color:#e65100; font-weight:bold;">✏️ Respuesta corregida según tu feedback</span>', unsafe_allow_html=True)
        
        st.markdown(answer)
        
        # ✅ MOSTRAR FUENTES AL FINAL DE LA RESPUESTA
        if sources:
            st.markdown('<div class="sources-container">', unsafe_allow_html=True)
            st.markdown("### 📚 Fuentes consultadas:")
            source_badges = " ".join([f'<span class="source-badge">📄 {source}</span>' for source in sources])
            st.markdown(source_badges, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Guardar respuesta completa con fuentes en el historial
    full_response = answer
    if metrics.get("corrected", False):
        full_response = f'<span class="feedback-indicator">✏️ Corregido</span><br>' + full_response
    if sources:
        source_list = ", ".join(sources)
        full_response += f'<br><br><div class="sources-container"><strong>📚 Fuentes:</strong> {source_list}</div>'
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.metrics = metrics
    
    
    st.rerun()

# ... [todo tu código anterior] ...

# ════════════════════════════════════════════════════════════════════════════
# 📝 FOOTER (ya existente en tu código)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Universidad del Valle • Grupo GUIA • ChatAcredita PRO v2.3<br>
    🌐 Avatar reducido (28px) + Señal visual clara de corrección
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 🔄 SCROLL AUTOMÁTICO 100% FUNCIONAL (NUEVO - AL FINAL ABSOLUTO)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<script>
function forceScrollToBottom() {
    const anchor = window.parent.document.getElementById('chat-bottom');
    if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
        return true;
    }
    const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
    if (messages.length > 0) {
        messages[messages.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
        return true;
    }
    const main = window.parent.document.querySelector('section.main');
    if (main) {
        main.scrollTop = main.scrollHeight;
        return true;
    }
    return false;
}
[100, 300, 600, 1000, 1500, 2000].forEach(delay => setTimeout(forceScrollToBottom, delay));
</script>
""", unsafe_allow_html=True)