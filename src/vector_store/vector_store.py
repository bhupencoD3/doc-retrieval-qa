from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore: FAISS = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """Create FAISS vectorstore from documents"""
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first")
        return self.retriever

    def save_vectorstore(self, path: str):
        Path(path).parent.mkdir(exist_ok=True)
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Vectorstore path {path} does not exist")
        # Warning: allow_dangerous_deserialization=True is a security risk
        # Only use with trusted FAISS indexes
        self.vectorstore = FAISS.load_local(
            path, embeddings=self.embedding, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})