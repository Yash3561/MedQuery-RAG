# app.py (v2.6 - FINAL with Unified State and Confidence Score)
import streamlit as st
from engine import MedQueryEngine

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="MedQuery-RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 2. CSS for Styling ---
st.markdown("""
    <style>
    .scrollable-container { max-height: 300px; overflow-y: auto; border: 1px solid #444; padding: 10px; border-radius: 5px; background-color: #1a1a1a; }
    .scrollable-container .source-item { border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 10px; }
    .scrollable-container .source-item:last-child { border-bottom: none; margin-bottom: 0; }
    .scrollable-container h4 { color: #00aaff; margin-top: 5px; font-size: 1em; }
    </style>
""", unsafe_allow_html=True)

# --- 3. Model & Engine Loading ---
@st.cache_resource
def load_engine():
    """Loads the MedQueryEngine once and caches it for the session."""
    return MedQueryEngine()

with st.spinner("Warming up the MedQuery Engine... This may take a moment."):
    engine = load_engine()

# --- 4. UNIFIED State Management ---
# We now use a SINGLE list of messages. Each message is a dictionary.
# This is the "single source of truth".
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to MedQuery-RAG. How can I help you?", "sources": None, "score": 0.0}
    ]

# --- 5. UI Function Definitions ---

def show_project_details():
    """Displays the project details using a stable st.container."""
    ### THIS IS THE FINAL FIX ###
    # We create a container that will only appear if the session_state flag is True.
    # This is a much more stable pattern than the experimental st.dialog.
    with st.container(border=True):
        st.header("Project Architecture")
        st.markdown("""
            ### MedQuery-RAG: An S-Tier AI Engineering Showcase
        
        This project is an end-to-end demonstration of a modern, production-grade AI system. It's designed to be a secure, verifiable, and high-performance knowledge engine, built entirely with open-source tools and deployed on a simulated, restricted HPC environment.
        
        ---
        
        #### Key Features & Demonstrated Skills:
        
        *   **End-to-End RAG Pipeline:** The core of the project is a sophisticated Retrieval-Augmented Generation system that mitigates LLM hallucination by grounding all responses in a verifiable knowledge base.
        *   **High-Performance Inference:** The backend is powered by **vLLM**, leveraging techniques like PagedAttention to achieve a **5-10x increase in inference throughput** compared to standard Hugging Face pipelines.
        *   **Scalable Data Engineering:**
            - The knowledge base was created using a **synthetic data generation pipeline**, where Llama 3 itself was used to author a comprehensive, 200+ entry medical encyclopedia.
            - This data is indexed into a **FAISS** vector store for lightning-fast semantic retrieval.
        *   **Advanced Precision Enhancement:** A **Cross-Encoder re-ranking model** (`ms-marco-MiniLM-L-6-v2`) is used as a secondary filter. This dramatically improves the signal-to-noise ratio of the context provided to the LLM, ensuring higher-quality and more accurate answers.
        *   **Professional UX/UI:** The entire system is wrapped in a responsive and interactive **Streamlit** web application, featuring a ChatGPT-style interface and transparent source-viewing capabilities.
        
        ---
        
        #### Technical Stack:
        
        *   **Core LLM:** `meta-llama/Llama-3.1-8B-Instruct`
        *   **Inference Engine:** `vLLM`
        *   **Embedding Model:** `BAAI/bge-large-en-v1.5`
        *   **Vector Store:** `FAISS` (Facebook AI Similarity Search)
        *   **Re-ranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
        *   **Data & ML Libraries:** `PyTorch`, `NumPy`, `Pandas`, `scikit-learn`
        *   **Web Framework:** `Streamlit`
        """)
        if st.button("Close", key="close_details"):
            st.session_state.show_details = False
            st.rerun()
    ### END OF FIX ###

def show_main_chat_interface():
    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        st.markdown(f"Welcome, **{st.session_state.user_name}**!")
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = [{"role": "assistant", "content": f"History cleared. Hello again, {st.session_state.user_name}! How can I assist you?", "sources": None, "score": 0.0}]
            st.rerun()
        if st.button("Project Details", key="project_details"):
            st.session_state.show_details = not st.session_state.get("show_details", False)
            st.rerun()
        st.markdown("---")
        st.info("*This is a technical demo and not a substitute for professional medical advice.*")
        st.markdown("---")
        st.markdown("Built by **Yash Chaudhary**")

    # --- Main Chat Area ---
    st.title("🩺 MedQuery-RAG")

    if st.session_state.get("show_details", False):
        show_project_details()

    # Display chat history from the single message list
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                score = message.get("score", 0.0)
                score_color = "green" if score > 0 else "orange" if score > -2.0 else "red"
                st.info(f"**Relevance Score:** :{score_color}[{score:.2f}] (Higher is better)")
                
                with st.expander("🧠 View Retrieved Sources"):
                    ### THIS IS THE FIRST FIX ###
                    html_parts = []
                    for doc in message["sources"]:
                        # Step 1: Do the replace operation first, outside the f-string
                        source_text_html = doc['text'].replace('\n', '<br>')
                        # Step 2: Now the f-string is clean
                        html_parts.append(f"""<div class="source-item"><h4>Source: {doc['metadata']['source_condition']}</h4><p>{source_text_html}</p></div>""")
                    sources_html = "".join(html_parts)
                    st.markdown(f'<div class="scrollable-container">{sources_html}</div>', unsafe_allow_html=True)
                    ### END OF FIX ###

    # Accept user input
    if prompt := st.chat_input("Ask a question about a medical condition..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None, "score": 0.0})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer, sources, raw_chunks, score = engine.answer_query(prompt, st.session_state.messages)
            st.markdown(answer)

            if raw_chunks:
                score_color = "green" if score > 0 else "orange" if score > -2.0 else "red"
                st.info(f"**Relevance Score:** :{score_color}[{score:.2f}] (Higher is better)")
                with st.expander("🧠 View Retrieved Sources"):
                    ### THIS IS THE SECOND FIX ###
                    html_parts = []
                    for doc in raw_chunks:
                        # Step 1: Do the replace operation first, outside the f-string
                        source_text_html = doc['text'].replace('\n', '<br>')
                        # Step 2: Now the f-string is clean
                        html_parts.append(f"""<div class="source-item"><h4>Source: {doc['metadata']['source_condition']}</h4><p>{source_text_html}</p></div>""")
                    sources_html = "".join(html_parts)
                    st.markdown(f'<div class="scrollable-container">{sources_html}</div>', unsafe_allow_html=True)
                    ### END OF FIX ###
        
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": raw_chunks, "score": score
        })

def show_welcome_form():
    """The initial welcome screen and name input form."""
    st.title("Welcome to MedQuery-RAG!")
    st.markdown("This is an AI-powered medical knowledge engine. Please enter your name to begin.")
    with st.form("welcome_form"):
        name = st.text_input("Your Name")
        if st.form_submit_button("Start Chatting"):
            if name:
                st.session_state.user_name = name
                # Initialize the single message list
                st.session_state.messages = [{"role": "assistant", "content": f"Hello, {name}! I'm ready to answer your questions. What would you like to know?", "sources": None, "score": 0.0}]
                st.rerun()
            else:
                st.error("Please enter your name.")

# --- 6. Main Application Logic ---
# This controls the initial view of the app.
if "user_name" not in st.session_state:
    show_welcome_form()
else:
    show_main_chat_interface()