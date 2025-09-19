from src.state.rag_state import RAGState

class SimpleRAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> dict:
        docs = self.retriever.invoke(state.question)
        return {"retrieved_docs": docs}

    def generate_answer(self, state: RAGState) -> dict:
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"""Answer the question based on the context
Context:
{context}
Question:
{state.question}"""
        response = self.llm.invoke(prompt)
        return {"answer": response.content, "retrieved_docs": state.retrieved_docs}
