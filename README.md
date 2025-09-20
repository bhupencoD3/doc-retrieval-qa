# Document Retrieval Question Answering (RAG with LangChain and LangGraph)

This repository implements a lightweight retrieval-augmented generation (RAG) system. The application is designed to support document ingestion from heterogeneous sources, embed and store the processed data in a vector database, and enable retrieval-based question answering through integration with large language models (LLMs). The system is accessible via a Streamlit interface and demonstrates both simple and agentic approaches to RAG using LangChain and LangGraph.

---

## System Capabilities

1. **Document ingestion**
   The system can process multiple types of input, including web URLs, directories of PDF files, and plain text documents. Documents are preprocessed and chunked to improve retrieval granularity.

2. **Vector storage and retrieval**
   Text chunks are embedded using OpenAI’s `text-embedding-3-small` model and stored in a FAISS vector database. FAISS provides efficient approximate nearest neighbor (ANN) search for scalable retrieval.

3. **Retrieval pipelines**
   Two complementary approaches are implemented:

   * **SimpleRAGNodes**: A direct pipeline where retrieved chunks are concatenated with the user’s query and passed to an LLM for answer synthesis.
   * **ReActNode**: A hybrid pipeline that first attempts retrieval from the local vector store and, in cases where the local documents provide insufficient evidence, falls back to an external knowledge source (Wikipedia). The Wikipedia integration is mediated by a ReAct-style agent built with LangGraph, allowing iterative reasoning and tool use.

4. **Summarization**
   To prevent verbose outputs, both pipelines incorporate summarization prompts that constrain the generated response to two or three sentences that directly answer the query.

5. **Frontend**
   A Streamlit-based application provides an interface for uploading documents, managing vector stores, and posing natural language queries.

6. **Persistence**
   Vector stores can be saved locally to enable repeated querying without reprocessing the entire corpus.

---

## Project Structure

```
ProjectRAG/
├── data/                   # Input sources (e.g. url.txt)
├── src/
│   ├── config/             # Application configuration
│   ├── document_ingestion/ # Document loading and preprocessing
│   ├── graph_builder/      # Graph-based pipelines (LangGraph)
│   ├── nodes/              # RAG nodes: SimpleRAGNodes and ReActNode
│   ├── state/              # Shared state definitions
│   └── vector_store/       # FAISS vector store utilities
├── streamlit_app.py        # Main Streamlit application
├── test_app.py             # Experimental scripts and prototyping
├── vectorstore/            # Persisted FAISS indexes
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/bhupencoD3/doc-retrieval-qa.git
cd doc-retrieval-qa
```

2. Create and activate a Python environment (example with Conda):

```bash
conda create -n rag_app python=3.11 -y
conda activate rag_app
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the OpenAI API key by creating a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

5. Launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

---

## Internal Workflows

1. **Document ingestion**
   The `DocumentProcessor` class orchestrates the ingestion of documents. After loading, the content is segmented into smaller units using recursive text splitting. This ensures that embedding captures semantically coherent chunks while remaining sufficiently fine-grained for retrieval.

2. **Vector store construction**
   The `VectorStore` class manages embeddings and storage. Documents are embedded with OpenAI models, and FAISS is used for efficient retrieval. The retriever interface abstracts vector similarity search and exposes methods to retrieve the top-k relevant chunks for a query.

3. **Answer generation**

   * In **SimpleRAGNodes**, retrieved chunks are combined with the query into a prompt. The LLM then produces an answer grounded in the local corpus.
   * In **ReActNode**, the system first attempts the same retrieval. If relevant documents are available, they are summarized. If not, a ReAct agent is constructed with access to a Wikipedia search tool. This agent performs tool-augmented reasoning and generates an answer by combining reasoning steps and external content.

4. **Summarization**
   Both nodes employ summarization prompts, ensuring responses are concise and directly address the user’s question. If retrieved or external content exceeds a length threshold, an additional summarization pass is invoked.

5. **Frontend interaction**
   The Streamlit interface allows interactive exploration. Users can upload new data sources, trigger ingestion, build vector stores, and query the system. Results are displayed alongside the retrieved context.

---

## Example Usage in Code

```python
from src.document_ingestion.document_processor import DocumentProcessor
from src.vector_store.vector_store import VectorStore
from src.nodes.nodes import SimpleRAGNodes
from src.nodes.react_node import RAGNodes
from src.state.rag_state import RAGState

# Ingest documents
processor = DocumentProcessor()
docs = processor.process_sources(["data/url.txt"])

# Build vector store
vs = VectorStore()
vs.create_vectorstore(docs)
retriever = vs.get_retriever()

# Simple pipeline
simple_nodes = SimpleRAGNodes(retriever, llm)
state = RAGState(question="What is LangGraph?")
result_simple = simple_nodes.generate_answer(state)
print("SimpleRAG answer:", result_simple["answer"])

# ReAct pipeline
react_nodes = RAGNodes(retriever, llm)
result_react = react_nodes.generate_answer(state)
print("ReAct answer:", result_react["answer"])
```

---

## Requirements

* Python 3.9 or higher
* OpenAI API key

Core libraries:

* langchain, langgraph
* faiss-cpu
* streamlit
* openai
* pypdf, beautifulsoup4

A complete list is provided in [requirements.txt](requirements.txt).

---

## Future Work

* Extend the ReAct pipeline into a multi-node LangGraph graph, enabling more complex reasoning chains beyond a single retrieval or fallback.
* Incorporate additional file formats (e.g., Word documents, Markdown).
* Replace the basic Streamlit interface with a more advanced conversational UI supporting multi-turn dialogue.
* Integrate monitoring and observability tools for model and retrieval performance analysis.

---

## License

This repository is distributed under the MIT License.
