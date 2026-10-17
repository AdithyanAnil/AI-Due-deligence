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
