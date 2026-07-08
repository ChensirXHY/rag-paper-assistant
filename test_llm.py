from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# 加载同目录下的.env文件
load_dotenv()

# 读取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model_name = os.getenv("DEEPSEEK_MODEL")

# 初始化DeepSeek
llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
    temperature=0.1
)

if __name__ == "__main__":
    #流式回复，思考一段话打出 一段话
    stream = llm.stream("用三句话介绍RAG检索增强生成技术")
    print("模型回复:")
    for chunk in stream:
        if chunk.content:
            print(chunk.content,end="",flush=True)