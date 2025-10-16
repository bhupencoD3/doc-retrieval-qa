# Document Retrieval Question Answering (RAG with LangChain and LangGraph)

This repository implements a lightweight retrieval-augmented generation (RAG) system. The application supports document ingestion from multiple sources, embeds and stores processed text in a **FAISS** vector database, and performs retrieval-based question answering using large language models (LLMs). It includes a **Streamlit** interface and demonstrates both simple and agentic RAG approaches with LangChain and LangGraph.

---

## **System Capabilities**

1. **Document Ingestion**
   Supports input from web URLs, directories of PDFs, and text documents. Data is preprocessed and chunked to enhance retrieval.

2. **Vector Storage and Retrieval**
   Uses OpenAI’s `text-embedding-3-small` model for embeddings and stores vectors in a **FAISS** database for efficient nearest neighbor search.

3. **Retrieval Pipelines**

   * **SimpleRAGNodes**: Directly concatenates retrieved chunks with user queries and sends them to an LLM for synthesis.
   * **ReActNode**: Combines local retrieval with a fallback Wikipedia agent (using LangGraph) for external reasoning.

4. **Graph-Based Pipeline (LangGraph)**
   The `GraphBuilder` constructs a **LangGraph StateGraph** that defines retrieval and generation nodes with explicit state handling.

5. **Summarization**
   Summarizes responses into concise, two-to-three-sentence answers.

6. **Frontend (Streamlit)**
   A web interface for document uploads, vector store management, and natural language querying.

7. **Persistence**
   FAISS indexes are saved locally for reuse without reprocessing.

8. **Configuration**
   Centralized in `src/config/config.py`, which loads environment variables (e.g., `OPENAI_API_KEY`) and manages chunk size, model, and retrieval settings.

---

## **Project Structure**

```
ProjectRAG/
├── data/
├── src/
│   ├── config/
│   │   └── config.py
│   ├── document_ingestion/
│   ├── graph_builder/
│   ├── nodes/
│   ├── state/
│   └── vector_store/
├── streamlit_app.py
├── test_app.py
├── vectorstore/
├── requirements.txt
└── README.md
```

---

## **Installation and Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/bhupencoD3/doc-retrieval-qa.git
   cd doc-retrieval-qa
   ```

2. Create and activate a virtual environment:

   ```bash
   conda create -n rag_app python=3.11 -y
   conda activate rag_app
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment:

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

5. Launch the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

---

## **Internal Workflows**

### **1. Document Ingestion**

The `DocumentProcessor` loads and preprocesses data, splitting it into coherent text chunks for embedding.

### **2. Vector Store Construction**

`VectorStore` manages embeddings and retrieval using FAISS. It supports top-k similarity search and persistent storage.

### **3. Answer Generation**

* **SimpleRAGNodes**: Combines query + retrieved chunks → LLM → summarized output.
* **ReActNode**: If local retrieval fails, a Wikipedia-powered ReAct agent (via LangGraph) is used.
* **GraphBuilder**: Defines a modular graph with nodes for retrieval and generation, enabling composable pipelines.

### **4. Summarization**

Keeps answers short and relevant using summarization prompts.

### **5. Frontend Interaction**

Streamlit UI for document ingestion, query answering, and displaying retrieved context.

### **6. Configuration Management**

Environment-driven configuration via `Config`. Adjusts chunk sizes, models, and retrieval parameters.

---

## **Example Usage**

```python
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vector_store.vector_store import VectorStore
from src.nodes.nodes import SimpleRAGNodes
from src.nodes.react_node import RAGNodes
from src.state.rag_state import RAGState
from src.graph_builder.graph_builder import GraphBuilder

llm = Config.get_llm()
processor = DocumentProcessor()
docs = processor.process_sources(["data/url.txt"])

vs = VectorStore()
vs.create_vectorstore(docs)
retriever = vs.get_retriever()

simple_nodes = SimpleRAGNodes(retriever, llm)
state = RAGState(question="What is LangGraph?")
result_simple = simple_nodes.generate_answer(state)
print(result_simple["answer"])

react_nodes = RAGNodes(retriever, llm)
print(react_nodes.generate_answer(state)["answer"])

graph_builder = GraphBuilder(retriever, llm)
graph = graph_builder.build()
print(graph_builder.run("Explain RAG with LangGraph.").answer)
```

---

## **Requirements**

* Python 3.9+
* OpenAI API key

Core libraries:

* langchain
* langgraph
* faiss-cpu
* streamlit
* openai
* pypdf
* beautifulsoup4
* python-dotenv

---

## **License**

Released under the MIT License.

---
