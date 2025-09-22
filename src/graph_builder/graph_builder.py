from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.react_node import RAGNodes

class GraphBuilder:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        builder = StateGraph(RAGState)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.set_entry_point("responder")
        builder.add_edge("responder", END)
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> RAGState:
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question, retrieved_docs=[], answer="")
        result = self.graph.invoke(initial_state)
        return result