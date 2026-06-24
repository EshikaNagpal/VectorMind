import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import os

# -------------------------
# Page Config (MUST be first)
# -------------------------

st.set_page_config(
    page_title="VectorMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Gemini Setup
# -------------------------

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False

# -------------------------
# Process PDF (unchanged)
# -------------------------

@st.cache_resource
def process_pdf(pdf_bytes):
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    chunk_embeddings = embedding_model.encode(chunks)
    vectors = np.array(chunk_embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return chunks, index

# -------------------------
# Custom CSS
# -------------------------

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App background ── */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #16181f !important;
    border-right: 1px solid #2a2d3a;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #9ca3b0;
    font-size: 13px;
}

/* ── Sidebar header ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #2a2d3a;
    margin-bottom: 20px;
}
.sidebar-logo-icon {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.sidebar-logo-text {
    font-size: 17px;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.3px;
}

/* ── Sidebar section labels ── */
.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4b5263;
    margin: 20px 0 8px 0;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1e2029 !important;
    border: 1.5px dashed #2e3141 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6c63ff !important;
}
[data-testid="stFileDropzoneInstructions"] {
    color: #6b7280 !important;
    font-size: 13px !important;
}

/* ── Stat cards ── */
.stat-card {
    background: #1e2029;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.stat-label {
    font-size: 11px;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.stat-value {
    font-size: 22px;
    font-weight: 600;
    color: #a78bfa;
    line-height: 1;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2e1a;
    border: 1px solid #2d4a2d;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #4ade80;
    margin-top: 6px;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #4ade80;
    border-radius: 50%;
}
.status-badge-idle {
    background: #1e2029;
    border: 1px solid #2a2d3a;
    color: #6b7280;
}
.status-dot-idle {
    background: #4b5263;
}

/* ── Dev mode toggle ── */
.stCheckbox label {
    color: #9ca3b0 !important;
    font-size: 13px !important;
}

/* ── Main area top padding ── */
.main-content {
    padding: 0 2rem;
    max-width: 820px;
    margin: 0 auto;
}

/* ── App header bar ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0 24px 0;
    border-bottom: 1px solid #1e2029;
    margin-bottom: 28px;
}
.app-header-title {
    font-size: 15px;
    font-weight: 600;
    color: #c4c6d0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.app-header-tag {
    font-size: 11px;
    background: #2a1f5e;
    color: #a78bfa;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
}

/* ── Welcome / Landing ── */
.welcome-container {
    text-align: center;
    padding: 60px 20px 40px;
}
.welcome-icon {
    font-size: 52px;
    margin-bottom: 20px;
    display: block;
}
.welcome-title {
    font-size: 38px;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.8px;
    margin-bottom: 12px;
    line-height: 1.2;
}
.welcome-title span {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.welcome-tagline {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 48px;
    line-height: 1.6;
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
}
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    max-width: 680px;
    margin: 0 auto 48px;
    text-align: left;
}
.feature-card {
    background: #16181f;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 18px 16px;
    transition: border-color 0.2s;
}
.feature-card:hover {
    border-color: #4b3fa0;
}
.feature-icon {
    font-size: 22px;
    margin-bottom: 10px;
    display: block;
}
.feature-title {
    font-size: 13px;
    font-weight: 600;
    color: #c4c6d0;
    margin-bottom: 4px;
}
.feature-desc {
    font-size: 12px;
    color: #6b7280;
    line-height: 1.5;
}
.onboarding-steps {
    background: #16181f;
    border: 1px solid #2a2d3a;
    border-radius: 14px;
    padding: 24px;
    max-width: 400px;
    margin: 0 auto;
    text-align: left;
}
.onboarding-title {
    font-size: 12px;
    font-weight: 600;
    color: #4b5263;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}
.onboarding-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}
.onboarding-step:last-child {
    margin-bottom: 0;
}
.step-num {
    width: 22px;
    height: 22px;
    background: #2a1f5e;
    color: #a78bfa;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 13px;
    color: #9ca3b0;
    line-height: 1.5;
}
.step-text strong {
    color: #c4c6d0;
    font-weight: 500;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    flex-direction: row-reverse !important;
}

/* ── Custom message wrappers ── */
.user-message-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 20px;
}
.user-bubble {
    background: #2a1f5e;
    border: 1px solid #3d2f8a;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    max-width: 78%;
    font-size: 14px;
    color: #ddd6fe;
    line-height: 1.6;
}

.assistant-message-wrap {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 20px;
    gap: 10px;
    align-items: flex-start;
}
.assistant-avatar {
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}
.assistant-card {
    background: #16181f;
    border: 1px solid #2a2d3a;
    border-radius: 4px 16px 16px 16px;
    padding: 16px 20px;
    max-width: 86%;
    font-size: 14px;
    color: #c4c6d0;
    line-height: 1.7;
}
.assistant-card h1, .assistant-card h2, .assistant-card h3 {
    color: #e8eaf0;
    margin-top: 12px;
    margin-bottom: 6px;
}
.assistant-card strong {
    color: #ddd6fe;
}
.assistant-card ol, .assistant-card ul {
    padding-left: 18px;
    margin: 8px 0;
}
.assistant-card li {
    margin-bottom: 6px;
    color: #b8bcc8;
}
.assistant-card p {
    margin-bottom: 10px;
}
.assistant-card p:last-child {
    margin-bottom: 0;
}

/* ── Section dividers inside answer ── */
.answer-section {
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #22253a;
}
.answer-section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}
.answer-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6c63ff;
    margin-bottom: 5px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #16181f !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 14px !important;
    color: #e8eaf0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e8eaf0 !important;
    font-size: 14px !important;
}

/* ── Source chunks expander (dev mode) ── */
.source-chunk {
    background: #1a1c24;
    border: 1px solid #2a2d3a;
    border-left: 3px solid #6c63ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #9ca3b0;
    line-height: 1.6;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.chunk-rank {
    font-size: 10px;
    font-weight: 700;
    color: #6c63ff;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.chunk-score {
    font-size: 10px;
    color: #4b5263;
    font-family: monospace;
}

/* ── Streamlit metric overrides ── */
[data-testid="stMetric"] {
    background: #1e2029;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 14px !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 24px !important;
    font-weight: 600 !important;
}

/* ── Success / info alerts ── */
[data-testid="stAlert"] {
    background: #1a2e1a !important;
    border: 1px solid #2d4a2d !important;
    border-radius: 10px !important;
    color: #4ade80 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4157; }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #1e2029;
    margin: 24px 0;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: #6c63ff !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🧠</div>
        <div class="sidebar-logo-text">VectorMind</div>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="sidebar-section-label">📄 Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        label_visibility="collapsed",
        help="Supported format: PDF"
    )

    # Document status
    if uploaded_file:
        st.markdown("""
        <div class="status-badge">
            <div class="status-dot"></div>
            Document ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-badge-idle">
            <div class="status-dot status-dot-idle"></div>
            No document loaded
        </div>
        """, unsafe_allow_html=True)

    # Stats (only when PDF loaded)
    if uploaded_file:
        pdf_bytes = uploaded_file.getvalue()
        chunks, index = process_pdf(pdf_bytes)

        st.markdown('<div class="sidebar-section-label" style="margin-top:24px;">📊 Statistics</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", len(chunks))
        with col2:
            st.metric("Vectors", index.ntotal)

        # File info
        size_kb = len(pdf_bytes) / 1024
        st.markdown(f"""
        <div class="stat-card" style="margin-top:8px;">
            <div class="stat-label">File size</div>
            <div style="font-size:14px; color:#c4c6d0; font-weight:500; margin-top:4px;">{size_kb:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)

    # Developer Mode
    st.markdown('<div class="sidebar-section-label" style="margin-top:24px;">⚙️ Settings</div>', unsafe_allow_html=True)
    st.session_state.dev_mode = st.checkbox(
        "Developer mode",
        value=st.session_state.dev_mode,
        help="Show retrieved source chunks and similarity scores for each answer."
    )

    # Footer
    st.markdown("""
    <div style="position:absolute; bottom:20px; left:16px; right:16px;">
        <div style="border-top:1px solid #2a2d3a; padding-top:14px;">
            <p style="font-size:11px; color:#3d4157; margin:0; text-align:center;">
                VectorMind · Powered by Gemini + FAISS
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Main Content Area
# -------------------------

# Pull chunks/index into scope if file is loaded
if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    chunks, index = process_pdf(pdf_bytes)

# Header bar
st.markdown("""
<div class="app-header">
    <div class="app-header-title">
        🧠 VectorMind
        <span class="app-header-tag">RAG · FAISS · Gemini</span>
    </div>
    <div style="font-size:12px; color:#3d4157;">Semantic PDF Search</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Welcome Screen (no PDF)
# -------------------------

if not uploaded_file:
    st.markdown("""
    <div class="welcome-container">
        <span class="welcome-icon">🧠</span>
        <div class="welcome-title">Chat with any PDF using<br><span>AI-powered search</span></div>
        <div class="welcome-tagline">
            Upload a document and ask questions in plain English.
            VectorMind finds the most relevant passages and generates
            precise, structured answers — instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <div class="feature-title">Semantic search</div>
            <div class="feature-desc">Finds meaning, not just keywords. Understands context across your document.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Instant answers</div>
            <div class="feature-desc">Gemini 2.5 Flash generates structured, accurate responses in seconds.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🗄️</span>
            <div class="feature-title">FAISS vectors</div>
            <div class="feature-desc">High-performance similarity search across thousands of document chunks.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🧩</span>
            <div class="feature-title">Smart chunking</div>
            <div class="feature-desc">Overlapping text segments preserve context across chunk boundaries.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Structured output</div>
            <div class="feature-desc">Every answer includes a definition, explanation, and key takeaway.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🔬</span>
            <div class="feature-title">Developer mode</div>
            <div class="feature-desc">Inspect retrieved chunks and similarity scores for full transparency.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Onboarding steps
    st.markdown("""
    <div class="onboarding-steps">
        <div class="onboarding-title">Get started in 3 steps</div>
        <div class="onboarding-step">
            <div class="step-num">1</div>
            <div class="step-text"><strong>Upload a PDF</strong> using the panel on the left — any document up to 200MB.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">2</div>
            <div class="step-text"><strong>Wait a moment</strong> while VectorMind indexes your document with FAISS.</div>
        </div>
        <div class="onboarding-step">
            <div class="step-num">3</div>
            <div class="step-text"><strong>Ask anything</strong> about the content in the chat box below.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# -------------------------
# Chat Interface (PDF loaded)
# -------------------------

# Render conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message-wrap">
            <div class="user-bubble">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message-wrap">
            <div class="assistant-avatar">🧠</div>
            <div class="assistant-card">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
question = st.chat_input("Ask a question about your document…")

if question:

    # Show user message immediately
    st.markdown(f"""
    <div class="user-message-wrap">
        <div class="user-bubble">{question}</div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": question})

    # ── RAG pipeline (UNCHANGED) ──

    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k=5)

    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an expert tutor.

Use ONLY the provided context.

Answer in this format:

1. Short Definition
2. Detailed Explanation
3. Key Takeaway

If the answer is not present in the context, say:

"I could not find that information in the PDF."

Context:
{context}

Question:
{question}
"""

    # Generate response with spinner
    with st.spinner(""):
        response = gemini_model.generate_content(prompt)
        answer = response.text

    # Format and display assistant message
    # Convert markdown to HTML-safe for our card (use st approach for md rendering)
    st.markdown(f"""
    <div class="assistant-message-wrap">
        <div class="assistant-avatar">🧠</div>
        <div class="assistant-card">
    """, unsafe_allow_html=True)

    st.markdown(answer)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Developer Mode: Source Chunks ──
    if st.session_state.dev_mode:
        with st.expander("🔬 Retrieved source chunks", expanded=False):
            st.markdown("""
            <div style="font-size:11px; color:#4b5263; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">
                Top 5 matches · ranked by L2 distance
            </div>
            """, unsafe_allow_html=True)

            for rank, idx in enumerate(indices[0]):
                score = distances[0][rank]
                chunk_text = chunks[idx]
                preview = chunk_text[:300] + "…" if len(chunk_text) > 300 else chunk_text

                st.markdown(f"""
                <div class="source-chunk">
                    <div class="chunk-header">
                        <span class="chunk-rank">Chunk #{rank + 1}</span>
                        <span class="chunk-score">L2 distance: {score:.4f}</span>
                    </div>
                    <div style="color:#9ca3b0; line-height:1.6;">{preview}</div>
                </div>
                """, unsafe_allow_html=True)