# 📄 Document Retrieval QA (RAG with LangChain + LangGraph)

A lightweight **Retrieval-Augmented Generation (RAG)** app built with **Streamlit**, combining **LangChain** and **LangGraph**.
This project ingests documents (URLs, PDFs, and text files), stores them in a FAISS vector database, and lets you query them with LLM-powered answers grounded in your data.

---

## ✨ Features

* 🔗 **Document ingestion**: Load from web URLs, PDF directories, or `.txt` files.
* 📑 **Chunking & preprocessing**: Recursive text splitting for better embedding + retrieval.
* 🧠 **Vector store**: Store & retrieve embeddings using **FAISS**.
* ⚡ **Retriever + LLM pipeline**: Powered by `langchain` + `langgraph`.
* 🎛 **Streamlit app**: Interactive frontend for querying your documents.
* 💾 **Persistence**: Save/load vector stores locally for reuse.

---

## 🏗️ Project Structure

```bash
ProjectRAG/
├── data/                   # Input sources (e.g. url.txt)
├── src/
│   ├── config/             # App configuration
│   ├── document_ingestion/ # Document loading + preprocessing
│   ├── graph_builder/      # Graph-based pipelines (LangGraph)
│   ├── nodes/              # RAG nodes (retrieval + answer generation)
│   ├── state/              # Shared state for pipeline
│   └── vector_store/       # FAISS vector store utilities
├── streamlit_app.py        # Main Streamlit app
├── test_app.py             # Test scripts / prototyping
├── vectorstore/            # Saved FAISS index
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/bhupencoD3/doc-retrieval-qa.git
cd doc-retrieval-qa
```

### 2. Create & activate environment

```bash
conda create -n rag_app python=3.11 -y
conda activate rag_app
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the app

```bash
streamlit run streamlit_app.py
```

---

## 🛠️ How It Works

1. **Document Ingestion**

   * `DocumentProcessor` loads sources (URLs, PDFs, `.txt`) and splits them into chunks.

2. **Vector Store**

   * `VectorStore` embeds docs with OpenAI’s `text-embedding-3-small` and stores them in **FAISS**.

3. **Retriever Node**

   * Retrieves top-`k` relevant chunks for a given query.

4. **Answer Generation Node**

   * `SimpleRAGNodes` combines the retrieved context + question, and uses an LLM to generate an answer.

5. **Streamlit Frontend**

   * Users interactively upload/query documents and see results.

---

## 📌 Example Usage

```python
from src.document_ingestion.document_processor import DocumentProcessor
from src.vector_store.vector_store import VectorStore
from src.nodes.nodes import SimpleRAGNodes
from src.state.rag_state import RAGState

# 1. Load & process docs
processor = DocumentProcessor()
docs = processor.process_sources(["data/url.txt"])

# 2. Create vector store
vs = VectorStore()
vs.create_vectorstore(docs)
retriever = vs.get_retriever()

# 3. Run retrieval + answer
rag_nodes = SimpleRAGNodes(retriever, llm)
state = RAGState(question="What is LangGraph?")
result = rag_nodes.generate_answer(state)
print(result["answer"])
```

---

## 📋 Requirements

* Python 3.9+
* OpenAI API key

Main libraries:

* `langchain`, `langgraph`, `streamlit`
* `faiss-cpu`, `openai`, `pypdf`, `beautifulsoup4`

Full list in [requirements.txt](requirements.txt).

---

## 📚 Roadmap

* [ ] Add **multi-node LangGraph pipeline** (e.g., React-style reasoning).
* [ ] Support **more file formats** (Word, Markdown).
* [ ] Enhance UI with chat-style interface in Streamlit.
* [ ] Integrate monitoring (Prometheus/Grafana).

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

* [LangChain](https://github.com/langchain-ai/langchain)
* [LangGraph](https://github.com/langchain-ai/langgraph)
* [Streamlit](https://streamlit.io/)
