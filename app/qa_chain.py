from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .config import Config

class QAChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # 自定义提示模板
        template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        尽量使用中文回答。

        上下文: {context}

        问题: {question}

        回答: """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建检索链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": Config.TOP_K}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )
    
    def answer(self, query):
        """生成答案"""
        try:
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]
            sources = [doc.metadata.get('source', 'Unknown') 
                      for doc in result["source_documents"]]
            
            return {
                "answer": answer,
                "sources": list(set(sources))  # 去重
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "抱歉，生成答案时出现错误。",
                "sources": []
            }