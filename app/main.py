
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
from .config import Config
from .embedding_manager import EmbeddingManager
from .qa_chain import QAChain
from .document_loader import DocumentLoader

app = FastAPI()

# 添加全局变量声明
qa_chain = None

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置常量
TEMP_UPLOAD_DIR = "temp_uploads"
ALLOWED_EXTENSIONS = {'.md'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 请求模型
class QuestionRequest(BaseModel):
    question: str

class DeleteFileRequest(BaseModel):
    filename: str

class FileResponse(BaseModel):
    filename: str
    size: int
    upload_time: str

# 确保临时目录存在
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/files/upload", response_model=FileResponse)
async def upload_file(file: UploadFile = File(...)):
    """上传单个文件到临时目录"""
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_ext}"
            )
        
        # 检查文件大小
        file_size = 0
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件过大: {file.filename}"
                    )
                buffer.write(chunk)
        
        return FileResponse(
            filename=file.filename,
            size=file_size,
            upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/files/delete")
async def delete_file(request: DeleteFileRequest):
    """从临时目录删除指定文件"""
    try:
        file_path = os.path.join(TEMP_UPLOAD_DIR, request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件不存在: {request.filename}"
            )
        
        os.remove(file_path)
        return {"message": f"文件已删除: {request.filename}"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/files/list", response_model=List[FileResponse])
async def list_files():
    """列出临时目录中的所有文件"""
    files = []
    for filename in os.listdir(TEMP_UPLOAD_DIR):
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        files.append(FileResponse(
            filename=filename,
            size=os.path.getsize(file_path),
            upload_time=datetime.fromtimestamp(
                os.path.getctime(file_path)
            ).strftime("%Y-%m-%d %H:%M:%S")
        ))
    return files

@app.post("/knowledge-base/generate")
async def generate_knowledge_base():
    """根据临时文件生成知识库"""
    try:
        # 检查临时目录是否为空
        if not os.listdir(TEMP_UPLOAD_DIR):
            raise HTTPException(
                status_code=400,
                detail="临时目录为空，请先上传文件"
            )
        
        # 加载文档
        loader = DocumentLoader()
        documents = loader.load_documents(TEMP_UPLOAD_DIR)
        
        # 创建向量存储
        global qa_chain
        vector_store = EmbeddingManager().create_vector_store(documents)
        qa_chain = QAChain(vector_store)
        
        return {"message": "知识库生成成功"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/query")
async def query(request: QuestionRequest):
    """知识库问答"""
    if not qa_chain:
        raise HTTPException(
            status_code=400,
            detail="知识库未初始化，请先生成知识库"
        )
    
    try:
        result = qa_chain.answer(request.question)
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/status")
async def get_status():
    """获取知识库状态"""
    return {
        "initialized": qa_chain is not None,
        "vector_store_exists": os.path.exists(Config.VECTOR_STORE_PATH),
        "temp_files_count": len(os.listdir(TEMP_UPLOAD_DIR))
    }

if __name__ == "__main__":
    # 如果存在向量存储，初始化 QA 链
    if os.path.exists(Config.VECTOR_STORE_PATH):
        vector_store = EmbeddingManager().load_vector_store(allow_dangerous_deserialization=True)
        qa_chain = QAChain(vector_store)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)