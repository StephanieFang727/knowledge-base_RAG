from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from typing import List
from pathlib import Path
from .config import Config

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            # separator="\n",
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def load_documents(self, data_dir: Path) -> List:
        """加载文档"""
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.md",
            show_progress=True
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def add_document(self, content: str, metadata: dict = None) -> List:
        """添加单个文档"""
        if metadata is None:
            metadata = {}
        texts = self.text_splitter.split_text(content)
        return [Document(page_content=t, metadata=metadata) for t in texts]