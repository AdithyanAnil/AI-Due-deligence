#app.py
import os
import shutil
import uuid
import atexit

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

from src.report import (
    collect_evidence,
    generate_report,
    create_pdf
)

from src.loader import load_uploaded_pdf
from src.processor import split_documents
from src.embeddings import create_embeddings
from src.vectorstore import create_vectorstore, create_retriever
# from src.rag import create_rag_chain
from src.rag_groq import create_rag_chain




# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Due Diligence Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# THEME / STYLES
# =========================

st.markdown(
    """
    <style>
    :root {
        --dd-bg: #020d1f;
        --dd-sidebar: #06162d;
        --dd-surface: #071b36;
        --dd-surface-2: #0a2345;
        --dd-surface-3: #0d2b53;
        --dd-border: #10477b;
        --dd-border-soft: #12345d;
        --dd-text: #f4f8ff;
        --dd-muted: #9cb8dc;
        --dd-accent: #00d9ff;
        --dd-blue: #087cf0;
        --dd-violet: #a446f5;
        --dd-success: #13e8b3;
    }

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
        }
    .stApp {
        background: radial-gradient(1000px 540px at 66% 18%, rgba(0, 105, 210, .09), transparent 65%), var(--dd-bg);
        color: var(--dd-text);
    }
    .block-container {
        max-width: none;
        padding: 5rem 1.4rem 2rem 19.3rem;
    }

    /* fixed left rail */
    section[data-testid="stSidebar"] {
        width: 282px !important;
        min-width: 282px !important;
        background: linear-gradient(180deg, #061b35 0%, var(--dd-sidebar) 58%, #041226 100%);
        border-right: 1px solid #123a68;
        box-shadow: 12px 0 36px rgba(0, 0, 0, .24);
    }
    section[data-testid="stSidebar"] > div { width: 282px !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0; }
    section[data-testid="stSidebar"] .block-container { padding: 1.35rem 1.25rem 1.6rem !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: var(--dd-text); letter-spacing: 0; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: var(--dd-muted) !important; }

    .brand-lockup { display: flex; align-items: center; gap: 13px; margin: 0 0 1.1rem; }
    .brand-mark, .hero-mark {
        display: grid; place-items: center; flex: 0 0 auto;
        color: white; font-weight: 900;
        background: linear-gradient(145deg, var(--dd-accent), #187ff5 56%, var(--dd-violet));
        box-shadow: 0 0 26px rgba(0, 191, 255, .28);
        clip-path: polygon(50% 0, 94% 24%, 94% 75%, 50% 100%, 6% 75%, 6% 24%);
    }
    .brand-mark { width: 50px; height: 54px; font-size: 23px; }
    .brand-title { color: var(--dd-text); font-size: 1.35rem; font-weight: 760; line-height: 1.1; }
    .brand-subtitle { color: var(--dd-muted); font-size: .74rem; margin-top: 5px; }
    .side-nav { display: grid; gap: 6px; margin: 0 0 1rem; }
    .side-nav-item { display: flex; align-items: center; gap: 12px; min-height: 42px; padding: 0 12px; border-radius: 8px; color: #c9dcf5; font-size: .88rem; font-weight: 600; }
    .side-nav-item.active { color: #fff; background: linear-gradient(90deg, #075a9e, #08499a); box-shadow: inset 0 0 0 1px rgba(17, 156, 255, .35); }
    .side-nav-icon { width: 22px; text-align: center; color: #91cfff; font-size: 1.05rem; }
    .side-label { margin: .85rem 0 .65rem; color: #9ac5ef; font-size: .67rem; text-transform: uppercase; font-weight: 700; }

    hr, div[data-testid="stDivider"] { border-color: var(--dd-border-soft) !important; margin: .8rem 0 !important; }
    section[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploaderDropzone"] {
        min-height: 64px; padding: .55rem !important; background: #082345;
        border: 1px dashed #0788d4; border-radius: 9px;
    }
    div[data-testid="stFileUploaderDropzone"] button { color: var(--dd-text) !important; border-color: var(--dd-border) !important; }
    div[data-testid="stFileUploaderFile"] { background: var(--dd-surface-2); border: 1px solid var(--dd-border-soft); border-radius: 8px; padding: .55rem; }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] { border-radius: 8px; background: rgba(0, 220, 175, .14); border: 1px solid rgba(0, 220, 175, .32); color: var(--dd-success); }

    .dd-stat { display: flex; justify-content: space-between; align-items: center; min-height: 42px; border: 1px solid var(--dd-border-soft); border-radius: 8px; background: var(--dd-surface-2); padding: 8px 12px; margin-bottom: 8px; }
    .dd-stat .k { color: var(--dd-muted); font-size: .67rem; text-transform: uppercase; font-weight: 650; }
    .dd-stat .v { color: var(--dd-text); font-size: .78rem; font-weight: 700; }
    .dd-stat .v.ok { color: var(--dd-success); }

    /* app chrome */
    header[data-testid="stHeader"] { background: rgba(2, 13, 31, .92); border-bottom: 1px solid var(--dd-border-soft); height: 62px; }
    .dd-topbar { position: fixed; z-index: 990; top: 0; left: 282px; right: 0; height: 62px; pointer-events: none; border-bottom: 1px solid var(--dd-border-soft); background: rgba(2, 13, 31, .86); }
    .dd-top-actions { position: absolute; right: 20px; top: 12px; display: flex; align-items: center; gap: 10px; color: var(--dd-muted); font-size: .76rem; }
    .dd-top-action { min-width: 74px; padding: 8px 14px; text-align: center; border: 1px solid var(--dd-border-soft); border-radius: 8px; background: #071a34; color: #dceaff; }
    .dd-top-action.primary { border-color: #168ef3; background: #087cf0; color: white; }

    /* content grid */
    .dd-header { position: relative; min-height: 164px; display: flex; align-items: center; overflow: hidden; border: 1px solid #108cc5; border-radius: 12px; padding: 24px 32px; background: radial-gradient(460px 160px at 18% 8%, rgba(0, 213, 255, .26), transparent 66%), linear-gradient(125deg, #075c93 0%, #0a377a 43%, #252090 100%); box-shadow: 0 14px 34px -26px rgba(0, 174, 255, .85); margin-bottom: 16px; }
    .dd-header::after { content: ""; position: absolute; width: 310px; height: 92px; right: -14px; bottom: -30px; border-top: 1px solid rgba(34, 208, 255, .65); border-radius: 50%; transform: rotate(-9deg); box-shadow: 0 -9px 0 -8px rgba(161, 70, 245, .85), 0 -20px 0 -19px rgba(31, 144, 255, .8); }
    .hero-mark { width: 91px; height: 98px; font-size: 38px; margin-right: 36px; }
    .dd-hero-copy { position: relative; z-index: 1; }
    .dd-header h1 { margin: 0; color: var(--dd-text); font-size: 1.9rem; line-height: 1.12; font-weight: 760; letter-spacing: 0; }
    .dd-header h1 span { color: var(--dd-accent); }
    .dd-header p { margin: 9px 0 0; color: #c7dcf6; font-size: .92rem; }
    .dd-pill { display: inline-block; margin-top: 14px; padding: 7px 15px; color: var(--dd-accent); border: 1px solid rgba(0, 217, 255, .52); border-radius: 999px; background: rgba(0, 116, 201, .30); font-size: .68rem; font-weight: 750; text-transform: uppercase; }

    .dd-card { border: 1px solid var(--dd-border); border-radius: 10px; background: linear-gradient(145deg, #071b36, #082044); padding: 19px 25px; margin: 0 0 16px; min-height: 244px; }
    .dd-card h4 { margin: 0 0 4px; color: var(--dd-text); font-size: 1.28rem; letter-spacing: 0; }
    .dd-card p { margin: 0 0 12px; color: var(--dd-muted); font-size: .83rem; }
    .dd-example { position: relative; border: 1px solid #15508a; border-radius: 10px; padding: 10px 42px 10px 14px; margin-top: 8px; color: #d9e8fb; background: linear-gradient(100deg, #0a2447, #0b294e); font-size: .79rem; line-height: 1.25; }
    .dd-example::after { content: "→"; position: absolute; right: 15px; top: 50%; transform: translateY(-50%); color: #9ec9ed; font-size: 1.05rem; }

    .feature-panel-title { display: none; }
    .feature-card { position: relative; min-height: 76px; border: 1px solid #174d83; border-radius: 10px; background: linear-gradient(120deg, #071d3a, #0a2047); padding: 14px 34px 12px 88px; margin: 4px 0 10px; box-shadow: none; transition: border-color .15s ease, background .15s ease; }
    .feature-card:hover, .feature-selected { border-color: #167fca !important; background: linear-gradient(120deg, #08274b, #121e56); }
    .feature-icon { position: absolute; left: 18px; top: 14px; width: 48px; height: 48px; border-radius: 50%; display: grid; place-items: center; font-size: 23px; color: white; }
    .feature-icon.document { background: linear-gradient(135deg, #08c8aa, #008c83); }
    .feature-icon.risk { background: linear-gradient(135deg, #9c45f4, #5f2dd1); }
    .feature-icon.business { background: linear-gradient(135deg, #04bfea, #0876df); }
    .feature-icon.evidence { background: linear-gradient(135deg, #a842ef, #6a31df); }
    .feature-name { color: var(--dd-text); font-size: .83rem; font-weight: 750; line-height: 1.2; }
    .feature-description { color: var(--dd-muted); font-size: .68rem; line-height: 1.42; margin-top: 5px; }
    .feature-arrow { position: absolute; right: 16px; top: 27px; color: #a7cff3; font-size: 20px; }
    .feature-mode { margin: 5px 0 14px; padding: 8px 11px; border: 1px solid rgba(0, 191, 255, .26); border-radius: 8px; background: rgba(0, 191, 255, .06); color: #b9d3ef; font-size: .70rem; line-height: 1.35; }

    div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--dd-border) !important; border-radius: 10px !important; background: linear-gradient(120deg, #071b36, #08244a); padding: 0 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] h3 { color: var(--dd-text); font-size: 1.04rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] p { color: var(--dd-muted); font-size: .76rem; }
    div[data-testid="stChatMessage"] { max-width: 920px; border: 1px solid var(--dd-border-soft); border-radius: 9px; background: var(--dd-surface); padding: 12px 14px; margin-bottom: 8px; }
    div[data-testid="stChatInput"] { background: #09244a; border: 1px solid #1b5a96; border-radius: 10px; }
    div[data-testid="stChatInput"] textarea { color: var(--dd-text); }

    .stButton > button, .stDownloadButton > button { min-height: 42px; border-radius: 8px; border: 1px solid #138ee0; background: linear-gradient(105deg, var(--dd-accent), var(--dd-blue) 55%, var(--dd-violet)); color: white; font-weight: 700; font-size: .80rem; box-shadow: none; transition: filter .12s ease, transform .12s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); filter: brightness(1.08); color: white; border-color: var(--dd-accent); }
    section[data-testid="stSidebar"] .stButton:last-child > button { background: transparent; color: var(--dd-accent); border-color: #0876bd; }

    div[data-testid="stStatusWidget"], details[data-testid="stExpander"] { border-radius: 8px; }
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

    @media (max-width: 1050px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .dd-topbar { left: 0; }
        .dd-header { min-height: 148px; padding: 20px; }
        .hero-mark { width: 70px; height: 76px; margin-right: 22px; }
    }
    @media (max-width: 720px) {
        .block-container { padding-top: 4.4rem; }
        .dd-top-actions .dd-top-action:not(.primary) { display: none; }
        .dd-header { align-items: flex-start; min-height: 220px; }
        .hero-mark { width: 56px; height: 61px; margin-right: 15px; }
        .dd-header h1 { font-size: 1.45rem; }
        .dd-header p { font-size: .8rem; }
        .dd-card { padding: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# SESSION
# =========================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.session_state.session_dir = os.path.join(
        "temp_data",
        st.session_state.session_id
    )

    os.makedirs(
        st.session_state.session_dir,
        exist_ok=True
    )


session_dir = st.session_state.session_dir


# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = "Document Analysis"


# =========================
# CLEANUP
# =========================

def cleanup():

    if os.path.exists(session_dir):

        try:
            shutil.rmtree(session_dir)

        except Exception:
            pass


atexit.register(cleanup)



# =========================
# AI FEATURE CARDS
# =========================

FEATURES = {
    "Document Analysis": {
        "icon": "📄",
        "class": "document",
        "description": "Extract key information and insights from your documents.",
        "prompt": "Analyze the uploaded documents and extract the most important financial, business, operational and organizational information. Answer using the retrieved document evidence."
    },
    "Risk Assessment": {
        "icon": "🔍",
        "class": "risk",
        "description": "Identify financial, operational and compliance risks.",
        "prompt": "Focus on risk assessment. Identify financial, operational, regulatory, compliance and business risks supported by the uploaded documents. Include supporting evidence and page numbers when available. Clearly distinguish reported risks from areas requiring further investigation."
    },
    "Business Insights": {
        "icon": "📈",
        "class": "business",
        "description": "Get data-driven insights and business summaries.",
        "prompt": "Focus on business insights. Analyze revenue trends, profitability, business segment performance, important business changes, growth opportunities and other significant business findings supported by the uploaded documents."
    },
    "Evidence Based": {
        "icon": "⚡",
        "class": "evidence",
        "description": "Every answer is grounded in source documents.",
        "prompt": "Use only information supported by the uploaded documents. For every important finding, provide the source document and page number when available. Do not use unsupported outside facts. If the information is unavailable, clearly state that it is not stated in the provided documents."
    }
}


def render_feature_cards():

    st.markdown(
        '<div class="feature-panel-title">AI Analysis</div>',
        unsafe_allow_html=True
    )

    for name, info in FEATURES.items():

        selected = st.session_state.selected_feature == name
        selected_class = " feature-selected" if selected else ""

        st.markdown(
            f"""
            <div class="feature-card{selected_class}">
                <div class="feature-icon {info['class']}">{info['icon']}</div>
                <div class="feature-arrow">›</div>
                <div class="feature-name">{name}</div>
                <div class="feature-description">{info['description']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            f"Use {name}",
            key=f"feature_{name}",
            use_container_width=True
        ):
            st.session_state.selected_feature = name
            st.rerun()

    selected = st.session_state.selected_feature

    st.markdown(
        f"""
        <div class="feature-mode">
            <b>Active mode:</b> {selected}<br>
            {FEATURES[selected]['description']}
        </div>
        """,
        unsafe_allow_html=True
    )

    return selected


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">◇</div>
            <div>
                <div class="brand-title">Due Diligence</div>
                <div class="brand-subtitle">AI-powered document analysis</div>
            </div>
        </div>
        <div class="side-nav">
            <div class="side-nav-item active"><span class="side-nav-icon">⌂</span>Home</div>
            <div class="side-nav-item"><span class="side-nav-icon">▤</span>Documents</div>
        </div>
        <div class="side-label">Upload documents</div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload annual reports, financial statements or other business documents."
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) selected"
        )

        if st.button(
            "⚙  Process Documents  →",
            use_container_width=True
        ):

            all_documents = []

            try:

                with st.status(
                    "Processing documents...",
                    expanded=True
                ):

                    st.write(
                        "📄 Loading PDF documents..."
                    )

                    for uploaded_file in uploaded_files:

                        documents = load_uploaded_pdf(
                            uploaded_file,
                            session_dir
                        )

                        all_documents.extend(
                            documents
                        )

                    st.write(
                        f"✓ {len(all_documents)} pages loaded"
                    )

                    st.write(
                        "✂️ Creating document chunks..."
                    )

                    texts = split_documents(
                        all_documents
                    )

                    st.write(
                        f"✓ {len(texts)} chunks created"
                    )

                    st.write(
                        "🧠 Creating embeddings..."
                    )

                    embeddings = create_embeddings()

                    st.write(
                        "✓ Embeddings created"
                    )

                    st.write(
                        "🗄️ Creating vector database..."
                    )

                    vectorstore = create_vectorstore(
                        texts,
                        embeddings,
                        session_dir
                    )

                    retriever = create_retriever(
                        vectorstore
                    )

                    st.write(
                        "✓ Vector database ready"
                    )

                    st.write(
                        "🤖 Connecting RAG assistant..."
                    )

                    rag_chain = create_rag_chain(
                        retriever
                    )

                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = rag_chain

                    st.session_state.processed = True

                    st.session_state.document_count = (
                        len(uploaded_files)
                    )

                st.success(
                    "Documents processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {str(e)}"
                )

    else:

        st.info(
            "Upload PDF documents to start."
        )

    st.divider()

    if st.session_state.processed:

        st.subheader("📌 Status")

        st.markdown(
            """
            <div class="dd-stat">
                <span class="k">Assistant</span>
                <span class="v ok">Ready</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="dd-stat">
                <span class="k">Documents</span>
                <span class="v">{st.session_state.document_count}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    if st.button(
        "▢  Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================
# MAIN AREA
# =========================
st.markdown(
    """
    <div class="dd-topbar">
        <div class="dd-top-actions">
            <span>☼</span>
            <span class="dd-top-action">⌘ &nbsp; Stop</span>
            <span class="dd-top-action primary">➤ &nbsp; Deploy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

main_col, feature_col = st.columns([3.15, 1.35], gap="medium")

with main_col:



    st.markdown(
        """
        <div class="dd-header">
            <div class="hero-mark">▤</div>
            <div class="dd-hero-copy">
                <h1>AI Due Diligence <span>Copilot</span></h1>
                <p>Ask questions about your uploaded financial and business documents.</p>
                <div class="dd-pill">Evidence-based document intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================
    # WELCOME MESSAGE
    # =========================

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="dd-card">
                <h4>Getting Started</h4>
                <p>Upload documents and ask questions to analyze financial
                performance, risks, and key business insights.</p>
                <div class="dd-example">“Summarise revenue and margin trends over the last three years.”</div>
                <div class="dd-example">“What are the main risk factors disclosed in these documents?”</div>
                <div class="dd-example">“List any related-party transactions or contingent liabilities.”</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    selected_feature = st.session_state.get(
        "selected_feature",
        "Document Analysis"
    )

    st.markdown(
        f"""
        <div class="feature-mode">
            <b>Analysis mode:</b> {selected_feature}
            &nbsp;•&nbsp;
            {FEATURES[selected_feature]["description"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # CHAT HISTORY
    # =========================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"],
            avatar="🧑‍💼" if message["role"] == "user" else "📊"
        ):

            st.markdown(
                message["content"]
            )


    # =========================
    # CHAT INPUT
    # =========================

    user_input = st.chat_input(
        "Ask a question about your documents..."
    )


    if user_input:

        if not st.session_state.processed:

            st.warning(
                "Please upload and process your documents first."
            )

            st.stop()


        # USER MESSAGE

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        with st.chat_message("user", avatar="🧑‍💼"):

            st.markdown(
                user_input
            )


        # ASSISTANT MESSAGE

        with st.chat_message("assistant", avatar="📊"):

            with st.spinner(
                "🔎 Searching documents..."
            ):

                try:

                    selected_feature = st.session_state.get(
                        "selected_feature",
                        "Document Analysis"
                    )

                    feature_instruction = FEATURES[selected_feature]["prompt"]

                    rag_question = f"""
                    You are operating in {selected_feature} mode.

                    {feature_instruction}

                    User question:
                    {user_input}

                    Important:
                    - Base the answer on the uploaded document evidence.
                    - Do not invent facts.
                    - Mention source/page information when available.
                    """

                    answer = (
                        st.session_state.rag_chain.invoke(
                            rag_question
                        )
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Error: {str(e)}"
                    )

                    st.error(
                        error_message
                    )

    st.divider()

    with st.container(border=True):

        report_text_col, report_action_col = st.columns([3, 2])

        with report_text_col:

            st.subheader("📄 Due Diligence Report")

            st.write(
                "Generate a detailed due diligence report "
                "from the uploaded documents."
            )

        with report_action_col:

            if st.session_state.rag_chain:

                if st.button(
                    "✦  Generate Due Diligence Report  →",
                    use_container_width=True
                ):

                    try:

                        with st.status(
                            "Generating due diligence report...",
                            expanded=True
                        ):

                            st.write(
                                "🔎 Collecting evidence from documents..."
                            )

                            retriever = st.session_state.retriever

                            evidence = collect_evidence(
                                retriever
                            )

                            st.write(
                                "🤖 Analyzing evidence with OpenAI..."
                            )

                            report = generate_report(
                                evidence
                            )

                            st.write(
                                "📄 Creating PDF report..."
                            )

                            report_path = os.path.join(
                                session_dir,
                                "AI_Due_Diligence_Report.pdf"
                            )

                            create_pdf(
                                report,
                                report_path
                            )

                        st.success(
                            "Due diligence report generated successfully!"
                        )

                        with open(
                            report_path,
                            "rb"
                        ) as pdf_file:

                            st.download_button(
                                label="⬇️ Download Due Diligence Report",
                                data=pdf_file,
                                file_name="AI_Due_Diligence_Report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                    except Exception as e:

                        st.error(
                            f"Report generation failed: {str(e)}"
                        )

            else:

                st.info(
                    "Process your documents first to generate the report."
                )

with feature_col:

    selected_feature = render_feature_cards()
#---------------------------------------------------------------------------------------------------------------

#embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings():
    embeddings=HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
#---------------------------------------------------------------------------------------------------------------

#loader.py
import os
from langchain_community.document_loaders import PyPDFLoader


def load_uploaded_pdf(uploaded_file, session_dir):

    upload_dir = os.path.join(
        session_dir,
        "uploads"
    )

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(
        upload_dir,
        uploaded_file.name
    )

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")

    with open(file_path, "wb") as file:
        file.write(file_bytes)

    if os.path.getsize(file_path) == 0:
        raise ValueError("PDF was saved as an empty file.")

    print("PDF loading...")

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents

#---------------------------------------------------------------------------------------------------------------


#processor.py


from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=['\n\n','\n']
    )
    texts=text_splitter.split_documents(documents)

    return texts
#---------------------------------------------------------------------------------------------------------------


#rag.py


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(retriever):

    llm = ChatOpenAI(
        model="gpt-5.6-luna",
        temperature=0.3,
        max_tokens=512
    )

    template = """
Answer the question based only on the following context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
Insufficient information in the uploaded documents.
"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
#---------------------------------------------------------------------------------------------------------------

#report.py

import os
import re
import html
from openai import OpenAI
from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ---------------------------------------------------------
# 0. UNICODE FONT REGISTRATION
# ---------------------------------------------------------
# FIX: the script never registered a real font, so every style used
# Helvetica. Helvetica's built-in encoding has no glyph for the Indian
# Rupee sign (\u20b9) or several other characters the LLM commonly
# outputs, so ReportLab silently replaced them with "\u25a0" boxes in
# the final PDF. We register a Unicode-capable TTF (DejaVu Sans, which
# does contain the Rupee sign) and use it everywhere instead. If it
# isn't installed on a given machine we fall back to Helvetica rather
# than crashing.

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

_UNICODE_FONT_CANDIDATES = [
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "C:\\Windows\\Fonts\\arialuni.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ),
    (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
]


def register_fonts():
    """Register a Unicode-capable font so currency symbols like Rupee
    render correctly. Sets the module-level FONT_REGULAR/FONT_BOLD
    names used by build_styles() and add_page_number(). Safe to call
    more than once; falls back to Helvetica if no candidate is found.
    """

    global FONT_REGULAR, FONT_BOLD

    for regular_path, bold_path in _UNICODE_FONT_CANDIDATES:

        if os.path.exists(regular_path):

            try:
                pdfmetrics.registerFont(TTFont("ReportUnicode", regular_path))

                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont("ReportUnicode-Bold", bold_path))
                    FONT_BOLD = "ReportUnicode-Bold"
                else:
                    FONT_BOLD = "ReportUnicode"

                FONT_REGULAR = "ReportUnicode"
                return

            except Exception:
                continue

    # No Unicode font found on this machine - keep the Helvetica
    # defaults rather than failing the whole report.


register_fonts()


# ---------------------------------------------------------
# 1. COLLECT EVIDENCE
# ---------------------------------------------------------

def collect_evidence(retriever):

    questions = [
        "What are the company's major financial achievements and financial performance?",
        "What are the company's major business segments and how are they performing?",
        "What are the company's major risks and risk mitigation measures?",
        "What are the company's major growth opportunities and future strategies?",
        "Are there any financial warning signs, red flags or areas requiring investigation?",
        "Are there any contradictions or inconsistencies in the reported information?",
        "What important information is missing, unclear or requires further investigation?"
    ]

    evidence = []

    for question in questions:

        docs = retriever.invoke(question)

        evidence.append(
            f"\nQUESTION: {question}\n"
        )

        for doc in docs:

            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Document")

            # LangChain PDF loaders often use zero-based page numbers.
            # Convert to human-readable page numbering when numeric.
            if isinstance(page, int):
                page = page + 1

            evidence.append(
                f"""
SOURCE: {source}
PAGE: {page}

{doc.page_content}
"""
            )

    return "\n".join(evidence)


# ---------------------------------------------------------
# 2. GENERATE REPORT CONTENT
# ---------------------------------------------------------

def generate_report(evidence):

    prompt = f"""
You are a senior due diligence analyst writing a formal report for a
reviewer who will act on it. Your job is accuracy, traceability and
clarity, not persuasion.

Base the report ONLY on the EVIDENCE at the end of this message.

=====================================================================
PART 1 - ANALYSIS RULES
=====================================================================

Before writing, silently work out: the company name, the reporting
periods covered, the currency and unit used (for example "INR crore"),
and whether each figure is standalone or consolidated. Use these
consistently throughout.

Grounding
- Never invent facts, figures, dates, page numbers or sources.
- Never fill in a missing value. If something is not in the evidence,
  write exactly: Not clearly stated in the provided documents.
- Do not draw conclusions the evidence does not support.
- Do not give investment advice, buy/sell/hold views or valuation views.
- Keep three things clearly separate: (a) reported facts, (b) potential
  concerns inferred from the facts, (c) items needing further
  investigation. Never present (b) or (c) as confirmed problems.

Numbers
- Copy reported figures exactly, with the currency symbol, unit and
  period label (for example "₹78,089 crore", "FY2026").
- Write every percentage with one decimal place (for example 41.1%).
- Any figure you derive yourself (growth %, margins, shares of total)
  must be computed from reported figures, double-checked, and marked
  "(Calculated)". Never mark a reported figure as calculated.
- Standalone and consolidated figures are different reporting scopes.
  Never compare them as if they were the same thing, and do not call
  a difference between them a contradiction unless the evidence shows
  a real conflict within the same scope and period.
- Compare like with like: same metric definition, same scope, same
  period. If definitions differ (for example "revenue from operations"
  vs "revenue including other income"), say so.

Citations
- Cite the page for every material fact, inline in plain parentheses,
  for example (p. 331) or (pp. 431-432). In tables use the Source
  column. Only cite pages that appear in the evidence.
- Never put a citation on a line by itself and never bold it.

Explicit vs inferred
- Risks the company itself states are "Reported by company".
- Risks you infer from the numbers are "Inferred". Label each one and
  never blur the two.

=====================================================================
PART 2 - OUTPUT FORMAT CONTRACT (strict, the output is machine parsed)
=====================================================================

The output is converted to a PDF by a program, so formatting mistakes
break the document. Follow these rules exactly.

Headings
- Use only these levels: "# " once for the title, "## " for the twelve
  numbered sections, "### " for sub-sections. Never use "####" or deeper.
- A heading is always on its own line, followed by a blank line.
- Never use a bold line as a substitute for a heading.
- Never use horizontal rules ("---", "***" or "___").

Tables
- Put a blank line before and after every table.
- Every row, including the header row and the separator row, is on
  its own single line that starts with "|" and ends with "|".
- Never put two rows on one line. Never split a row across lines.
- Never put a line break, "<br>", or a bullet list inside a cell.
- Every row must have exactly the same number of cells as the header.
- Keep each cell under about 25 words. If a point needs more, split it
  into another row.
- Use at most about 12 rows per table. If there is more, keep the most
  material rows.
- Use exactly the column headers given below, in the same order.
- The separator row uses "---" for text columns and "---:" for numeric
  columns, with the same number of cells as the header.
- If a required table has no supporting evidence at all, still output
  the table with a single row whose first cell is
  Not clearly stated in the provided documents. and "-" in the rest.

Text
- One bullet per line, starting with "- ". Write each bullet as a
  single line with no manual line wrapping. No nested bullets.
- Numbered items are single lines starting with "1. ", "2. ", etc.
- Paragraphs are 1-3 sentences, written as a single line each.
- Use **bold** sparingly and only inside a sentence, never for a whole
  line.
- No emojis, no HTML, no code fences, no block quotes.
- Do not add anything before the title or after the last section.

=====================================================================
PART 3 - REPORT TEMPLATE
=====================================================================

Start with this exact structure. Replace text in angle brackets.

# Due Diligence Report: <Company Name>

One or two sentences stating the documents and periods covered, the
currency unit, and that the report relies only on the evidence supplied.

## 1. Executive Summary

Write 5-7 bullets. Each bullet states one key finding with its
figures, period and page. Cover: overall performance, main growth
driver, main concern, and the most important information gap.

| Key Area | Finding | Evidence / Source |
|---|---|---|

Give 5-8 rows covering performance, profitability, segments, risks,
red flags and data gaps.

## 2. Company Overview

Two or three short paragraphs or bullets: what the business does, its
main segments, reporting scope (standalone / consolidated), and
notable organisational facts stated in the evidence.

| Area | Reported Information | Source |
|---|---|---|

## 3. Financial Analysis

Use a "### 3.1 Consolidated financial performance" and a
"### 3.2 Standalone financial performance" sub-section when both are
available; otherwise use a single sub-section for the scope available.
Each sub-section has one table:

| Metric | Current Period | Previous Period | Change/Growth | Observation | Source Page |
|---|---:|---:|---:|---|---|

Include every available metric among: revenue, operating profit,
profit before tax, profit after tax, finance costs, exceptional items,
margins, assets, liabilities, debt, cash, cash flow, EPS. Do not
include a metric that has no figure in the evidence. Mark derived
growth and margin values "(Calculated)".

After the tables, add "### 3.3 Key financial observations" with 3-6
bullets interpreting the numbers (trends, margin movement, finance
cost movement, exceptional items, what cannot be assessed and why).

## 4. Business Segment Analysis

Use one table per reporting scope available, in sub-sections
"### 4.1 <scope> segment performance":

| Segment | Current Revenue | Previous Revenue | Revenue Growth | Current Profit | Previous Profit | Key Observation | Source Page |
|---|---:|---:|---:|---:|---:|---|---|

Then, for each material segment, add a "### 4.x <Segment name>"
sub-section with 2-4 bullets: what changed, how profit moved relative
to revenue, share of total (Calculated), and what the evidence does
not explain.

## 5. Key Business Insights

### 5.1 Reported facts

Bullets containing only facts directly stated in the evidence.

### 5.2 Potential concerns

Bullets of evidence-based concerns. Each must say what in the
evidence prompted it and must not be worded as a confirmed problem.

### 5.3 Areas requiring investigation

Bullets of issues that need more information.

| Insight | Type | Evidence | Source Page |
|---|---|---|---|

Type must be exactly one of: Reported fact, Potential concern,
Requires investigation. Include the most material 6-10 insights.

## 6. Growth Opportunities

| Opportunity | Supporting Evidence | Potential Business Impact | Limitations / Unknowns | Source Page |
|---|---|---|---|---|

Only opportunities supported by the evidence. Do not invent targets,
forecasts or expected returns. If a strategy or target is not stated,
say so in the Limitations column.

## 7. Major Risks

| Risk | Basis | Evidence | Potential Impact | Reported Mitigation | Further Investigation | Source Page |
|---|---|---|---|---|---|---|

Basis must be exactly "Reported by company" or "Inferred". If no
mitigation is stated, write Not clearly stated in the provided
documents. in that cell.

## 8. Warnings and Potential Red Flags

| Warning / Potential Red Flag | Supporting Evidence | Why It Requires Attention | Confirmed or Requires Investigation | Source Page |
|---|---|---|---|---|

Use "Confirmed" only when the evidence itself establishes the point.
Otherwise use "Requires investigation". After the table, add one
sentence stating whether the evidence shows any confirmed misstatement,
qualification, going-concern issue or enforcement action.

## 9. Contradictions or Inconsistencies

| Topic | Information A | Information B | Source Pages | Assessment |
|---|---|---|---|---|

Assessment must be one of: Confirmed contradiction, Reporting-scope
difference, Presentation or definition difference, Requires
reconciliation. If there is no confirmed contradiction, add one
sentence saying so explicitly.

## 10. Missing or Unclear Information

| Missing / Unclear Information | Why It Matters | Available Evidence | Suggested Investigation |
|---|---|---|---|

Cover financial, operational, risk and governance gaps. Prioritise the
most material gaps first (for example debt, cash flow, liquidity,
nature of exceptional items, reconciliation of reporting scopes).

## 11. Questions for Further Investigation

| # | Investigation Question | Reason | Relevant Source/Page |
|---:|---|---|---|

Give 10-20 questions, numbered 1, 2, 3 in the first column, grouped in
order of importance. Each question must trace to a specific gap or
concern found above. Do not also repeat them as a separate list.

## 12. Overall Due Diligence Observations

Use these four sub-sections, each with 2-5 bullets:

### 12.1 Evidence-based observations
### 12.2 Strengths supported by the documents
### 12.3 Areas requiring further investigation
### 12.4 Limitations of the available evidence

End with one sentence: No investment recommendation is made.

=====================================================================
PART 4 - FINAL SELF-CHECK (do silently, do not print)
=====================================================================

Before you answer, verify:
1. Every table row is on its own line and has the same number of cells
   as its header.
2. There are no horizontal rules, no "####" headings, no bold-only
   lines, no citations on their own line.
3. Every derived number is marked "(Calculated)" and is arithmetically
   correct.
4. No figure, page number or fact appears that is not in the evidence.
5. Standalone and consolidated figures are never mixed in one row.
6. The response contains only the report.

EVIDENCE:
{evidence}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


# ---------------------------------------------------------
# 3. PDF STYLES
# ---------------------------------------------------------

def build_styles():

    styles = getSampleStyleSheet()

    return {

        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=FONT_BOLD,
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            spaceAfter=8
        ),

        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=20
        ),

        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=19,
            spaceBefore=15,
            spaceAfter=9,
            textColor=colors.HexColor("#17365D"),
            keepWithNext=True
        ),

        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2F5597"),
            keepWithNext=True
        ),

        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=13,
            spaceAfter=6
        ),

        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=13,
            leftIndent=13,
            firstLineIndent=-7,
            spaceAfter=4
        ),

        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.2,
            leading=9,
            textColor=colors.white
        ),

        "table_cell": ParagraphStyle(
            "TableCell",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.1,
            leading=9,
            spaceAfter=0
        ),

        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#666666")
        )
    }


# ---------------------------------------------------------
# 4. MARKDOWN HELPERS
# ---------------------------------------------------------

def clean_markdown(text):

    text = text.strip()

    # Remove markdown bold/italic markers while preserving text.
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*", r"<i>\1</i>", text)

    # Escape ampersands that are not already entities.
    text = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;)",
        "&amp;",
        text
    )

    text = text.replace("<br>", "<br/>")

    return text


def paragraph_text(text, style):

    text = clean_markdown(text)

    return Paragraph(text, style)


def is_table_separator(line):

    cells = [c.strip() for c in line.strip().strip("|").split("|")]

    if not cells:
        return False

    # FIX: was "-{3,}" (required 3+ dashes). Some models emit shorter
    # separators such as "|-|-|"; loosened to 1+ so real separator
    # rows aren't missed and the table that follows isn't dropped.
    return all(
        re.fullmatch(r":?-+:?", c.replace(" ", ""))
        for c in cells
    )


def parse_table(lines, start_index):

    header_line = lines[start_index].strip()
    separator_line = lines[start_index + 1].strip()

    headers = [
        c.strip()
        for c in header_line.strip("|").split("|")
    ]

    column_count = len(headers)

    rows = []

    i = start_index + 2

    while i < len(lines):

        line = lines[i].strip()

        if not line or "|" not in line:
            break

        # FIX: the LLM sometimes runs two or more data rows together on
        # one line with no newline between them, joined as
        # "...| 553 crore || Eyecare | 907 crore ...". That used to
        # leave us with more cells than headers on that line, and the
        # code silently truncated to the first `column_count` cells -
        # discarding every row after the first (this is why segments
        # like "Others" were vanishing from the segment table).
        #
        # A squashed boundary always shows up as a literal "||" (the
        # closing pipe of one row butted against the opening pipe of
        # the next, with nothing in between), whereas a genuine empty
        # cell is written as "| |" with a space. So we un-squash by
        # inserting a line break at every "||" before splitting into
        # cells, recovering each row instead of merging/dropping it.
        for sub_line in line.replace("||", "|\n|").split("\n"):

            sub_line = sub_line.strip()

            if not sub_line or "|" not in sub_line:
                continue

            cells = [
                c.strip()
                for c in sub_line.strip("|").split("|")
            ]

            if len(cells) < column_count:
                cells += [""] * (column_count - len(cells))
            elif len(cells) > column_count:
                cells = cells[:column_count]

            rows.append(cells)

        i += 1

    return headers, rows, i


def make_table(headers, rows, styles):

    data = []

    data.append([
        Paragraph(clean_markdown(h), styles["table_header"])
        for h in headers
    ])

    for row in rows:
        data.append([
            Paragraph(clean_markdown(cell), styles["table_cell"])
            for cell in row
        ])

    column_count = len(headers)

    # A4 usable width with 15 mm margins.
    usable_width = 180 * mm

    if column_count <= 4:
        col_widths = [usable_width / column_count] * column_count
    else:
        # Give the last/observation columns more room.
        base = usable_width / column_count
        col_widths = [base] * column_count

        for index, header in enumerate(headers):
            h = header.lower()

            if any(
                word in h
                for word in [
                    "observation",
                    "evidence",
                    "impact",
                    "investigation",
                    "explanation",
                    "information",
                    "question"
                ]
            ):
                col_widths[index] = base * 1.45

        total = sum(col_widths)
        col_widths = [
            width * usable_width / total
            for width in col_widths
        ]

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#17365D")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#B7B7B7")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F5F7FA")
                ]
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    return table


# ---------------------------------------------------------
# 5. HEADER / FOOTER
# ---------------------------------------------------------

def add_page_number(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(colors.HexColor("#D9E1F2"))
    canvas.line(
        15 * mm,
        12 * mm,
        width - 15 * mm,
        12 * mm
    )

    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))

    canvas.drawString(
        15 * mm,
        7 * mm,
        "AI Due Diligence Report"
    )

    canvas.drawRightString(
        width - 15 * mm,
        7 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ---------------------------------------------------------
# 6. CREATE PROFESSIONAL PDF
# ---------------------------------------------------------

def create_pdf(report_text, output_path):

    styles = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Due Diligence Report",
        author="AI Due Diligence Copilot",
        subject="Due Diligence Analysis"
    )

    story = []

    # Cover/title area.
    story.append(
        Spacer(1, 20 * mm)
    )

    story.append(
        Paragraph(
            "AI DUE DILIGENCE REPORT",
            styles["title"]
        )
    )

    story.append(
        Paragraph(
            "Evidence-Based Business and Financial Analysis",
            styles["subtitle"]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    story.append(
        Paragraph(
            "<b>Report basis:</b> Analysis generated only from the "
            "documents and retrieved evidence supplied to the system.",
            styles["body"]
        )
    )

    story.append(
        Spacer(1, 10 * mm)
    )

    # FIX (structure): give the cover its own page instead of letting
    # body text run on underneath it.
    story.append(PageBreak())

    # FIX (structure): track whether we've reached a top-level "## N."
    # section yet, so each one after the first starts on a fresh page
    # instead of the whole report reading as one unbroken wall of text.
    first_top_level_section_seen = False

    # FIX (structure): a heading that lands at the very bottom of a
    # page with its content pushed to the next page reads as broken
    # structure. We hold each heading back until we know what follows
    # it, then emit heading+content together via KeepTogether so a
    # heading is never left dangling alone at a page break.
    pending_heading = None

    def flush_pending(with_flowables=None):
        nonlocal pending_heading
        if pending_heading is not None:
            group = [pending_heading] + (with_flowables or [])
            story.append(KeepTogether(group))
            pending_heading = None
        elif with_flowables:
            story.extend(with_flowables)

    # Report content.
    lines = report_text.splitlines()

    i = 0

    while i < len(lines):

        raw_line = lines[i].strip()

        if not raw_line:
            flush_pending()
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Ignore an LLM-generated duplicate title.
        if raw_line.lower() in {
            "# ai due diligence report",
            "ai due diligence report"
        }:
            i += 1
            continue

        # Markdown headings.
        if raw_line.startswith("### "):
            heading = raw_line[4:].strip()

            flush_pending()
            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h2"]
            )

            i += 1
            continue

        if raw_line.startswith("## "):
            heading = raw_line[3:].strip()

            flush_pending()

            if first_top_level_section_seen:
                story.append(PageBreak())

            first_top_level_section_seen = True

            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h1"]
            )

            i += 1
            continue

        if raw_line.startswith("# "):
            heading = raw_line[2:].strip()

            flush_pending()
            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h1"]
            )

            i += 1
            continue

        # Markdown table.
        if (
            "|" in raw_line
            and i + 1 < len(lines)
            and is_table_separator(lines[i + 1])
        ):

            headers, rows, next_index = parse_table(
                lines,
                i
            )

            if headers:
                table = make_table(
                    headers,
                    rows,
                    styles
                )

                flush_pending([
                    Spacer(1, 4),
                    table,
                    Spacer(1, 10)
                ])

            i = next_index
            continue

        # "Source: ..." captions read better as a small muted note
        # directly under the table/paragraph they refer to.
        if raw_line.lower().startswith("source:"):

            flush_pending([
                Paragraph(
                    clean_markdown(raw_line),
                    styles["small"]
                )
            ])

            i += 1
            continue

        # Markdown bullet.
        if raw_line.startswith("- "):

            content = raw_line[2:].strip()

            flush_pending([
                Paragraph(
                    "\u2022 " + clean_markdown(content),
                    styles["bullet"]
                )
            ])

            i += 1
            continue

        # Numbered list.
        if re.match(r"^\d+\.\s+", raw_line):

            content = re.sub(
                r"^\d+\.\s+",
                "",
                raw_line
            )

            # Keep investigation questions compact.
            flush_pending([
                Paragraph(
                    clean_markdown(raw_line),
                    styles["bullet"]
                )
            ])

            i += 1
            continue

        # Bold-only subsection text.
        if (
            raw_line.startswith("**")
            and raw_line.endswith("**")
            and raw_line.count("**") == 2
        ):

            heading = raw_line[2:-2].strip()

            flush_pending()
            pending_heading = Paragraph(
                f"<b>{html.escape(heading)}</b>",
                styles["h2"]
            )

            i += 1
            continue

        # Normal paragraph.
        flush_pending([
            Paragraph(
                clean_markdown(raw_line),
                styles["body"]
            )
        ])

        i += 1

    # Flush any heading left pending at the very end of the document.
    flush_pending()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return output_path
#---------------------------------------------------------------------------------------------------------------

#vectorstore.py
import os

from langchain_chroma import Chroma


def create_vectorstore(
    texts,
    embeddings,
    session_dir
):

    chroma_dir = os.path.join(
        session_dir,
        "chroma_db"
    )

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=chroma_dir
    )

    return vectorstore


def create_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )
#---------------------------------------------------------------------------------------------------------------