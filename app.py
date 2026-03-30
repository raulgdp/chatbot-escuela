# app.py - ChatAcredita PRO: Multiagente + Subida de documentos + OpenRouter corregido
import streamlit as st
import os, time, json, unicodedata, base64, uuid, re, io
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from rank_bm25 import BM25Okapi
import tempfile

# ════════════════════════════════════════════════════════════════════════════
# 🔐 CONFIGURACIÓN DE TESSERACT (PARA PROCESAMIENTO DE DOCUMENTOS)
# ════════════════════════════════════════════════════════════════════════════
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    import cv2
    from PIL import Image
    
    # Verificar disponibilidad de Tesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    st.sidebar.success("✅ Tesseract OCR disponible")
except Exception as e:
    st.sidebar.warning(f"⚠️ Tesseract no disponible: {str(e)[:50]}")
    st.sidebar.info("💡 Para procesar tablas, instala Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")

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
# ✅ CORREGIDO: URL sin espacios + modelo válido
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = get_secret("OPENAI_API_BASE", "https://openrouter.ai/api/v1").strip()  # ✅ Sin espacios

# ✅ MODELO VÁLIDO Y DISPONIBLE EN OPENROUTER
DEFAULT_MODEL = "meta-llama/llama-3.1-70b-instruct"  # ✅ Gratuito y estable

try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    # Validación rápida
    _ = client.models.list()
    st.sidebar.success(f"✅ OpenRouter conectado ({DEFAULT_MODEL})")
except Exception as e:
    st.sidebar.error(f"❌ Error OpenRouter: {str(e)[:80]}")
    st.sidebar.info("""
    🔑 Verifica en Settings → Secrets:
    • OPENAI_API_KEY = sk-or-v1-...
    • OPENAI_API_BASE = https://openrouter.ai/api/v1 (sin espacios)
    """)

# Conexión a Qdrant Cloud
try:
    qdrant = QdrantClient(
        url=get_secret("QDRANT_URL", "").strip(),
        api_key=get_secret("QDRANT_API_KEY", "").strip()
    )
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        st.error(f"❌ Colección '{COLLECTION_NAME}' no encontrada")
        st.stop()
    
    # Crear colección de feedback si no existe
    if FEEDBACK_COLLECTION not in [c.name for c in collections]:
        from qdrant_client.models import VectorParams, Distance
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
# 📤 PROCESAMIENTO DE DOCUMENTOS SUBIDOS (CORREGIDO)
# ════════════════════════════════════════════════════════════════════════════
import fitz  # PyMuPDF
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_uploaded_document(pdf_bytes, filename):
    """Procesa PDF subido con el mismo pipeline que embeddings-T.py"""
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        # Extraer texto con PyMuPDF4LLM
        doc = fitz.open(tmp_path)
        all_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = pymupdf4llm.to_markdown(
                doc, 
                pages=[page_num],
                show_progress=False,
                page_chunks=False,
            )
            
            # Detectar y procesar tablas con Tesseract si está disponible
            has_visual = (
                "figura" in page_text.lower() or
                "imagen" in page_text.lower() or
                "tabla" in page_text.lower() or
                "cuadro" in page_text.lower() or
                len(page.get_images()) > 0 or
                len([b for b in page.get_text("dict").get("blocks", []) if b.get("lines")]) > 15
            )
            
            if has_visual and TESSERACT_AVAILABLE:
                try:
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, 
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thresh = cv2.dilate(thresh, kernel, iterations=1)
                    
                    custom_config = (
                        r'--oem 3 --psm 6 -l spa+eng '
                        r'--dpi 300 '
                        r'-c preserve_interword_spaces=1 '
                        r'-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;()[]{}%$€#@/\\-_ñáéíóúüÁÉÍÓÚÜ°"'
                    )
                    
                    table_text = pytesseract.image_to_string(thresh, config=custom_config)
                    if table_text.strip():
                        table_text = normalize_text(table_text)
                        page_text += f"\n\n[TABLA PÁGINA {page_num + 1}]\n{table_text}"
                except Exception as e:
                    st.warning(f"⚠️ Error Tesseract en página {page_num+1}: {str(e)[:50]}")
            
            all_text += f"\n\n--- Página {page_num + 1} ---\n\n" + page_text
        
        doc.close()
        os.unlink(tmp_path)
        
        # Chunking estructural
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n## ", "\n\n### ", "\n\n#### ",
                "\n\n[TABLA ", "\n\n[IMAGEN ", "\n\n[FIGURA ",
                "\n\n|", "\n\n", "\n", " ", ""
            ],
            is_separator_regex=False,
        )
        
        chunks = splitter.split_text(all_text)
        valid_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 100:
                continue
            if (re.search(r'\|[^\n]*$', chunk) and not re.search(r'\|\s*$', chunk)) or \
               chunk.endswith("[TABLA") or chunk.endswith("[IMAGEN"):
                continue
            valid_chunks.append(chunk)
        
        return valid_chunks, [filename] * len(valid_chunks)
        
    except Exception as e:
        st.error(f"❌ Error procesando documento: {str(e)[:100]}")
        return [], []

def add_chunks_to_qdrant(chunks, sources):
    """Añade nuevos chunks a la colección existente en Qdrant Cloud"""
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
# 🔍 BÚSQUEDA HÍBRIDA
# ════════════════════════════════════════════════════════════════════════════
def hybrid_search(query, use_feedback=False):
    collection = FEEDBACK_COLLECTION if use_feedback else COLLECTION_NAME
    
    try:
        emb = embedder.encode([query], normalize_embeddings=True)[0]
        results = qdrant.query_points(
            collection_name=collection,
            query=emb.tolist(),
            limit=TOP_K,
            with_payload=True
        ).points
        
        return [r.payload["text"] for r in results if r.payload]
    except Exception as e:
        st.warning(f"⚠️ Error búsqueda: {str(e)[:50]}")
        return []

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
            docs_original = hybrid_search(query, use_feedback=False)
            docs_feedback = hybrid_search(query, use_feedback=True)
            context = "\n\n".join(docs_original + docs_feedback)[:4000]
        else:
            docs = hybrid_search(query, use_feedback=False)
            context = "\n\n".join(docs)[:4000]
        
        answer = self.answer_agent.run(query, context)
        latency = round(time.time() - start, 2)
        
        # Corregir si es retroalimentación
        if intent == "retroalimentacion" and last_answer:
            try:
                r = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{
                        "role": "user", 
                        "content": f"Corrige: {last_answer} basado en: {query}"
                    }],
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
                st.sidebar.success("✅ Retroalimentación guardada")
                return corrected, {"latency": latency, "intent": intent}
            except:
                pass
        
        return answer, {"latency": latency, "intent": intent}

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
# 📥 MANEJO DE INPUT DEL USUARIO
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
    
    # Mostrar avatar de "pensando"
    thinking = st.empty()
    with thinking.container():
        st.markdown(f"""
        <div class="thinking-avatar">
            <img src="data:image/webp;base64,{avatar_base64}" class="avatar-img">
            <span>🧠 Analizando...</span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    
    # Generar respuesta
    answer, metrics = rag_system.run(prompt, last_answer)
    thinking.empty()
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.metrics = metrics
    
    # Scroll automático
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
# 📝 FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Universidad del Valle • Grupo GUIA • ChatAcredita PRO v2.1<br>
    🌐 Sistema Multiagente con Persistencia de Feedback
</div>
""", unsafe_allow_html=True)