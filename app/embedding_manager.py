from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import torch
from .config import Config
from langchain_community.embeddings.openai import OpenAIEmbeddings
import os


class EmbeddingManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def create_vector_store(self, documents):
        """创建向量存储"""
        vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        # 保存向量存储
        vector_store.save_local(Config.VECTOR_STORE_PATH)
        return vector_store
    
    def load_vector_store(self, allow_dangerous_deserialization=False):
        """加载现有向量存储"""
        if os.path.exists(Config.VECTOR_STORE_PATH):
            return FAISS.load_local(
                Config.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
        return None