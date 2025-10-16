import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vector_store.vector_store import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

VECTORSTORE_PATH = Path("vectorstore/faiss_index")

st.set_page_config(page_title="📚 ReAct RAG Search", page_icon="🔍", layout="centered")
st.markdown("""
<style>
.answer-box {padding: 15px; border-radius: 12px; font-size:16px; line-height:1.5; background-color:#2b2b2b; color:#fff;}
.source-doc-box {padding: 10px; border-radius:10px; font-size:14px; line-height:1.4; background-color:#1e1e1e; color:#f1f1f1;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Session state
# ------------------------------
def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []

# ------------------------------
# Initialize RAG
# ------------------------------
@st.cache_resource
def initialize_rag():
    st.write("🚀 Initializing RAG system...")
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    vector_store = VectorStore()

    if VECTORSTORE_PATH.exists():
        st.write("📂 Loading existing vectorstore...")
        vector_store.load_vectorstore(str(VECTORSTORE_PATH))
        documents_count = len(vector_store.vectorstore.index_to_docstore_id)
    else:
        st.write("⚡ Processing and embedding documents...")
        docs = doc_processor.process_sources(Config.DEFAULT_URLS)
        if not docs:
            st.error("Failed to load any documents. Check source URLs.")
            return None, 0, llm
        vector_store.create_vectorstore(docs)
        VECTORSTORE_PATH.parent.mkdir(exist_ok=True)
        vector_store.save_vectorstore(str(VECTORSTORE_PATH))
        documents_count = len(docs)
        st.write(f"✅ {documents_count} document chunks embedded!")

    graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
    graph_builder.build()
    return graph_builder, documents_count, llm

# ------------------------------
# Main App
# ----------------------
def main():
    init_session_state()
    st.title("📚 ReAct RAG Document Search")
    st.markdown("💬 Ask questions about the loaded documents")

    with st.sidebar:
        st.header("⚙️ Settings")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            st.caption(f"Documents loaded: {st.session_state.num_chunks}")
            st.caption(f"LLM: {st.session_state.llm_name}")
        else:
            st.warning("System not initialized")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []

    if not st.session_state.initialized:
        with st.spinner("⚡ Loading documents and initializing RAG..."):
            result = initialize_rag()
            if result is None:
                return
            rag_system, num_chunks, llm = result
            st.session_state.rag_system = rag_system
            st.session_state.num_chunks = num_chunks
            st.session_state.llm_name = str(llm)
            st.session_state.initialized = True
            st.success(f"🚀 System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    with st.form("search_form"):
        question = st.text_input("Enter your question:", placeholder="🔎 e.g., What is the main idea of the document?")
        submit = st.form_submit_button("Search")

        if submit and question and st.session_state.rag_system:
            with st.spinner("🔍 Running ReAct agent..."):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time

                answer = result.get("answer", "No answer generated")
                retrieved_docs = result.get("retrieved_docs", [])

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "time": elapsed_time
                })

                st.markdown("### ✅ Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                if retrieved_docs:
                    with st.expander("📂 Source Documents"):
                        for idx, doc in enumerate(retrieved_docs, start=1):
                            meta = doc.metadata if hasattr(doc, "metadata") else {}
                            title = meta.get("source", f"Document {idx}")
                            # Use a unique key for each text_area based on index
                            st.text_area(
                                label=f"📄 {title}",
                                value=(doc.page_content[:300] + "..." if hasattr(doc, "page_content") else ""),
                                height=120,
                                disabled=True,
                                key=f"source_doc_{idx}"  # Unique key for each text_area
                            )
                else:
                    st.info("No documents retrieved for this query.")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕘 Past Queries")
        for h in reversed(st.session_state.history):
            st.markdown(f"**Q:** {h['question']}")
            st.markdown(f"**A:** {h['answer']}")
            st.caption(f"⏱️ Time taken: {h['time']:.2f}s")

if __name__ == "__main__":
    main()