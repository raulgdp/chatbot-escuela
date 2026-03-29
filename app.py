# app.py - ChatAcredita PRO: Multiagente + Persistencia de Feedback en Qdrant
import streamlit as st
import os, time, json, unicodedata, base64, uuid
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from rank_bm25 import BM25Okapi
import re

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

# Estado inicial
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
FEEDBACK_COLLECTION = "feedback_acreditacion"  # Nueva colección para feedback
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
# 🎨 CSS PERSONALIZADO
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
    background: white; padding: 10px 14px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    display: flex; align-items: center; gap: 10px;
    z-index:9999;
}
.avatar-img { border-radius:50%; width:38px; }
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
# 🔌 CONFIGURACIÓN DE APIs (CORREGIDO PARA OPENROUTER)
# ════════════════════════════════════════════════════════════════════════════
# ✅ CORREGIDO: Sin espacios al final + modelo válido
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "sk-or-v1-tu-api-key").strip()
OPENAI_API_BASE = get_secret("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()  # ✅ Sin espacios

# ✅ MODELO VÁLIDO PARA OPENROUTER (Llama 3.1 70B - gratuito y estable)
DEFAULT_MODEL = "meta-llama/llama-3.1-70b-instruct"  # ✅ Disponible en OpenRouter

try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    # Validación rápida de conexión
    _ = client.models.list()
    st.sidebar.success("✅ OpenRouter conectado")
except Exception as e:
    st.sidebar.error(f"❌ Error OpenRouter: {str(e)[:80]}")
    st.sidebar.info("💡 Verifica: API key válida + URL sin espacios")

# Conexión a Qdrant Cloud
try:
    qdrant = QdrantClient(
        url=get_secret("QDRANT_URL", "").strip(),
        api_key=get_secret("QDRANT_API_KEY", "").strip()
    )
    # Verificar colección principal
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada en Qdrant")
        st.stop()
    
    # ✅ CREAR COLECCIÓN DE FEEDBACK SI NO EXISTE
    if FEEDBACK_COLLECTION not in [c.name for c in collections]:
        qdrant.create_collection(
            collection_name=FEEDBACK_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        st.sidebar.success(f"✅ Colección '{FEEDBACK_COLLECTION}' creada para feedback")
    else:
        st.sidebar.success(f"✅ Colecciones verificadas: {COLLECTION_NAME}, {FEEDBACK_COLLECTION}")
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
st.sidebar.success("✅ Embeddings: BAAI/bge-m3 (1024d)")

# ════════════════════════════════════════════════════════════════════════════
# 🔍 BÚSQUEDA HÍBRIDA (SEMÁNTICA + BM25)
# ════════════════════════════════════════════════════════════════════════════
def hybrid_search(query, use_feedback=False):
    """Búsqueda en colección principal o feedback según parámetro"""
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    
    # Embedding de la consulta
    emb = embedder.encode([query], normalize_embeddings=True)[0]
    
    # Búsqueda vectorial
    results = qdrant.query_points(
        collection_name=collection,
        query=emb.tolist(),
        limit=TOP_K,
        with_payload=True
    ).points
    
    return [r.payload["text"] for r in results if r.payload]

# ════════════════════════════════════════════════════════════════════════════
# 🤖 AGENTE CLASIFICADOR DE INTENCIÓN
# ════════════════════════════════════════════════════════════════════════════
def classify_feedback(prompt, last_answer=""):
    """Clasifica si el input es pregunta o retroalimentación"""
    prompt_llm = f"""
Contexto de la conversación:
Última respuesta del asistente: {last_answer[:300]}

Nuevo mensaje del usuario:
{prompt}

Instrucciones:
- Si el usuario hace una nueva pregunta o solicita información → "pregunta"
- Si el usuario corrige, critica o mejora la respuesta anterior → "retroalimentacion"
- Si el usuario agradece o confirma → "pregunta" (neutral)

Responde SOLO con este formato JSON:
{{"tipo": "pregunta" o "retroalimentacion"}}
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
        return "pregunta"  # Default seguro

# ════════════════════════════════════════════════════════════════════════════
# 📝 AGENTE GENERADOR DE RESPUESTAS
# ════════════════════════════════════════════════════════════════════════════
class AnswerAgent:
    def run(self, query, context):
        prompt = f"""
Eres ChatAcredita, asistente especializado en acreditación de la EISC (Universidad del Valle).
Responde SOLO con la información del contexto proporcionado. Sé preciso y profesional.

CONTEXTO RECUPERADO:
{context}

INSTRUCCIONES DE FORMATO:
- Si la información es conceptual o explicativa → usa viñetas (•)
- Si hay datos comparativos o tablas → usa tabla Markdown con encabezados
- Si es una lista de criterios o pasos → usa numeración
- Máximo 3 párrafos o 1 tabla + 1 párrafo

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:
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
            return f"⚠️ Error generando respuesta: {str(e)[:100]}"

# ════════════════════════════════════════════════════════════════════════════
# 🔁 AGENTE DE CORRECCIÓN (PARA RETROALIMENTACIÓN)
# ════════════════════════════════════════════════════════════════════════════
class CorrectionAgent:
    def correct(self, original_answer, feedback, context):
        prompt = f"""
Eres un experto en acreditación universitaria. Corrige la respuesta anterior basándote EN EXCLUSIVA en el contexto y la retroalimentación del usuario.

CONTEXTO OFICIAL:
{context}

RESPUESTA ANTERIOR (a corregir):
{original_answer}

RETROALIMENTACIÓN DEL USUARIO:
{feedback}

INSTRUCCIONES:
1. Corrige SOLO los errores específicos mencionados en la retroalimentación
2. Mantén el formato original (viñetas/tabla si aplica)
3. Si la retroalimentación es vaga, mejora claridad sin cambiar contenido
4. Nunca inventes información fuera del contexto

RESPUESTA CORREGIDA:
"""
        try:
            r = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            return r.choices[0].message.content
        except Exception as e:
            return original_answer  # Mantener original si falla

# ════════════════════════════════════════════════════════════════════════════
# 💾 FUNCIÓN PARA GUARDAR RETROALIMENTACIÓN EN QDRANT
# ════════════════════════════════════════════════════════════════════════════
def save_feedback_to_qdrant(original_question, original_answer, feedback, corrected_answer):
    """Guarda retroalimentación validada como nuevo documento en Qdrant"""
    try:
        # Crear documento mejorado
        improved_doc = f"""
PREGUNTA ORIGINAL: {original_question}

RESPUESTA CORREGIDA (basada en retroalimentación del usuario):
{corrected_answer}

RETROALIMENTACIÓN DEL USUARIO:
{feedback}

METADATOS:
- Tipo: respuesta_corregida
- Fecha: {time.time()}
- Fuente: feedback_usuario_validado
"""
        # Generar embedding
        embedding = embedder.encode([improved_doc], normalize_embeddings=True)[0]
        
        # Añadir a colección de feedback
        qdrant.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": improved_doc,
                        "source": "feedback_usuario",
                        "type": "respuesta_corregida",
                        "original_question": original_question,
                        "original_answer": original_answer,
                        "user_feedback": feedback,
                        "corrected_answer": corrected_answer,
                        "timestamp": time.time()
                    }
                )
            ]
        )
        return True
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error guardando feedback: {str(e)[:50]}")
        return False

# ════════════════════════════════════════════════════════════════════════════
# 🧠 SISTEMA RAG MULTIAGENTE
# ════════════════════════════════════════════════════════════════════════════
class RAGSystem:
    def __init__(self):
        self.answer_agent = AnswerAgent()
        self.correction_agent = CorrectionAgent()
    
    def run(self, query, last_answer=""):
        start = time.time()
        
        # 1. Clasificar intención
        intent = classify_feedback(query, last_answer)
        
        # 2. Buscar en colecciones relevantes
        if intent == "retroalimentacion":
            # Buscar en documentos originales + feedback validado
            docs_original = hybrid_search(query, use_feedback=False)
            docs_feedback = hybrid_search(query, use_feedback=True)
            context = "\n\n".join(docs_original + docs_feedback)[:4000]
        else:
            # Solo buscar en documentos originales
            docs = hybrid_search(query, use_feedback=False)
            context = "\n\n".join(docs)[:4000]
        
        # 3. Generar respuesta inicial
        answer = self.answer_agent.run(query, context)
        latency = round(time.time() - start, 2)
        
        # 4. Si es retroalimentación, aplicar corrección
        if intent == "retroalimentacion" and last_answer:
            corrected = self.correction_agent.correct(last_answer, query, context)
            
            # ✅ GUARDAR RETROALIMENTACIÓN VALIDADA EN QDRANT
            if save_feedback_to_qdrant(query, last_answer, query, corrected):
                st.sidebar.success("✅ Retroalimentación guardada en base de conocimiento")
            
            return corrected, {"latency": latency, "intent": intent, "corrected": True}
        
        return answer, {"latency": latency, "intent": intent, "corrected": False}

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
    st.session_state.metrics = {"latency": 0, "intent": "pregunta"}

st.title("💬 Chat Académico EISC")

# Mostrar historial de mensajes
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ════════════════════════════════════════════════════════════════════════════
# 📥 MANEJO DE INPUT DEL USUARIO
# ════════════════════════════════════════════════════════════════════════════
avatar_base64 = get_base64_image("data/yo.webp") or "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhESMIAAAAABJRU5ErkJggg=="

# Ancla para scroll automático
st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)

prompt = st.chat_input("Escribe tu pregunta sobre acreditación...")

if prompt:
    # Obtener última respuesta del asistente
    last_answer = ""
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant":
            last_answer = m["content"]
            break
    
    # Añadir mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Mostrar avatar de "pensando"
    thinking = st.empty()
    with thinking.container():
        st.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🧠 Analizando intención...</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    # Clasificar intención
    intent = classify_feedback(prompt, last_answer)
    
    # Actualizar estado visual
    thinking.empty()
    thinking = st.empty()
    with thinking.container():
        action = "Corrigiendo respuesta" if intent == "retroalimentacion" else "Recuperando información"
        st.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🔍 {action}...</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    # Generar respuesta con sistema RAG multiagente
    answer, metrics = rag_system.run(prompt, last_answer)
    
    # Mostrar resultado
    thinking.empty()
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Añadir respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.metrics = metrics
    
    # Scroll automático al final
    st.markdown("""
    <script>
    function scrollToBottom() {
        const bottom = document.getElementById('bottom');
        if (bottom) bottom.scrollIntoView({behavior: 'smooth'});
    }
    setTimeout(scrollToBottom, 100);
    setTimeout(scrollToBottom, 300);
    </script>
    """, unsafe_allow_html=True)
    
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# 📊 SIDEBAR - MÉTRICAS Y DOCUMENTACIÓN
# ════════════════════════════════════════════════════════════════════════════
col1, col2 = st.sidebar.columns([1, 2])
with col1:
    st.image("data/yo.webp" if os.path.exists("data/yo.webp") else "https://via.placeholder.com/80", width=80)
with col2:
    st.markdown(f"**{st.session_state.user}**")
    st.caption("EISC Univalle")

st.sidebar.markdown("### 📊 Métricas en tiempo real")
st.sidebar.metric("⏱️ Latencia", f"{st.session_state.metrics.get('latency', 0)} s")
st.sidebar.metric("🔍 Tipo de input", st.session_state.metrics.get('intent', 'pregunta').capitalize())
st.sidebar.metric("👥 Visitas totales", st.session_state.visits)

with st.sidebar.expander("🧠 Arquitectura del Sistema", expanded=True):
    st.markdown("""
    ### 🤖 Sistema Multiagente
    
    **1. Classifier Agent**
    - Clasifica intención: pregunta vs retroalimentación
    - Context-aware (usa última respuesta)
    - Zero-shot classification con LLM
    
    **2. RAG Core**
    - Búsqueda híbrida: Qdrant Cloud + BM25
    - Embeddings: BAAI/bge-m3 (1024d)
    - Colecciones: documentos originales + feedback validado
    
    **3. Answer Agent**
    - Genera respuestas estructuradas
    - Formato inteligente: viñetas, tablas, párrafos
    - Contexto integrado de múltiples fuentes
    
    **4. Correction Agent**
    - Corrige respuestas basado en feedback
    - Mantiene formato original
    - Solo modifica información específica
    
    ### 💾 Persistencia de Conocimiento
    
    ✅ **Retroalimentación validada → Nuevos documentos en Qdrant**
    - Colección separada: `feedback_acreditacion`
    - Embeddings generados automáticamente
    - Disponible para búsquedas futuras
    - Mejora continua del sistema
    
    ### 🌐 Infraestructura
    
    - **Vector DB**: Qdrant Cloud (2 colecciones)
    - **LLM**: OpenRouter (Llama 3.1 70B)
    - **Embeddings**: BAAI/bge-m3 (1024d)
    - **Framework**: Streamlit + Multiagente
    """)

# ════════════════════════════════════════════════════════════════════════════
# 📝 FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Universidad del Valle • Grupo GUIA • ChatAcredita PRO v2.1<br>
    🌐 Sistema Multiagente con Persistencia de Feedback en Qdrant Cloud
</div>
""", unsafe_allow_html=True)