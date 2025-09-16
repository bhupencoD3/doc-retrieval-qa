from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


from typing import Union,List
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFDirectoryLoader,
    TextLoader
)

class DocumentProcessor:
    """"Handles documents loading and processing"""
    def __init__(self,chunk_size:int=500,chunk_overlap:int=50):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_url(self,url:str) -> List[Document]:
        """"Loads documents from urls"""
        loader=WebBaseLoader(url)
        return loader.load()
    
    def load_pdf_dir(self,directory:Union[str,Path]) -> List[Document]:
        """"Loads documents from all PDFs inside a directory"""
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_text(self,file_path:Union[str,Path]) -> List[Document]:
        """"Loads documents from Txt files"""
        loader=TextLoader(str(file_path),encoding='utf-8')
        return loader.load()
    
    def load_pdf_(self,file_path:Union[str,Path]) -> List[Document]:
        """"Loads documents from all PDFs"""
        loader=PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self,sources:List[str]) -> List[Document]:
        """Loads documents from URLs,PDF directories or Txt files"""
        docs:List[Document]=[]
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_url(src))

            path=Path("data")
            if path.is_dir():
                docs.extend(self.load_pdf_dir(path))
            elif path.suffix.lower()=='.txt':
                docs.extend(self.load_text(path))
            else:
                raise ValueError(
                    f"Unsupported source type:{src}."
                    "Use URL,.txt or PDF directory"
                )
        return docs
    
    def split_documents(self,documents:List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
    
    def process_url(self,urls:List[str])->List[Document]:
        docs=self.load_documents(urls)
        return self.split_documents(docs)


