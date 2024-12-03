# 使用具体版本标签的基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 首先复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install -r requirements.txt

# 让 Python 能够在容器中找到 app 模块
ENV PYTHONPATH=/app

# 然后复制其他项目文件
COPY . .

# 暴露端口
EXPOSE 8001

# 运行应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]