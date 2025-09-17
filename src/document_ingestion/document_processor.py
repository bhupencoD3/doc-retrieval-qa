from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Union, List
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader, TextLoader

class DocumentProcessor:
    """Handles document loading and processing"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_url(self, url: str) -> List[Document]:
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            return []

    def load_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        try:
            loader = PyPDFDirectoryLoader(str(directory))
            return loader.load()
        except Exception as e:
            print(f"Error loading PDF directory {directory}: {e}")
            return []

    def load_text(self, file_path: Union[str, Path]) -> List[Document]:
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            return loader.load()
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []

    def load_documents(self, sources: List[str]) -> List[Document]:
        """Loads documents from URLs, PDF directories, or text files"""
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_url(src))
            else:
                src_path = Path(src)
                if src_path.is_dir():
                    docs.extend(self.load_pdf_dir(src_path))
                elif src_path.suffix.lower() == ".txt":
                    docs.extend(self.load_text(src_path))
                else:
                    print(f"Unsupported source type: {src}. Use URL, .txt, or PDF directory")
                    continue
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[str]) -> List[Document]:
        docs = self.load_documents(sources)
        if not docs:
            print("No documents loaded from sources")
            return []
        return self.split_documents(docs)