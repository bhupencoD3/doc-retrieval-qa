from typing import TypedDict, List
from langchain.schema import Document

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str