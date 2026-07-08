# import warnings
# from langchain_core._api.deprecation import LangChainDeprecationWarning
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import os
from dotenv import load_dotenv
# ==================== 修正后的导入（全部避免弃用警告）====================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# 加载 .env
load_dotenv()
# ------------------- 配置参数 -------------------
PDF_FOLDER = "./papers"
CHROMA_DB_DIR = "./chroma_db"
EMBED_MODEL_NAME = "./local_models/BAAI/bge-large-zh-v1.5"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
# -----------------------------------------------
# 1. 加载 PDF（新版写法，无弃用警告）
print("正在加载PDF文献...")
loader = DirectoryLoader(
    PDF_FOLDER,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
docs = loader.load()
print(f"共加载 {len(docs)} 页")
# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"切分成 {len(chunks)} 个文本块")
# 3. 嵌入模型
print("加载嵌入模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name = EMBED_MODEL_NAME,
    model_kwargs = {"device": "cuda"},   # 无显卡则写 "cpu"
    encode_kwargs={"normalize_embeddings": True}
)
# 4. 构建/加载向量库
if not os.path.exists(CHROMA_DB_DIR):
    print("创建向量库...")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
else:
    print("加载已有向量库...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
# 5. 初始化 DeepSeek API 模型
print("连接 DeepSeek API...")
llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=DEEPSEEK_MODEL,
    temperature=0.1
)
# 6. Prompt 模板
template = """你是一个学术文献助手。请仅根据以下文献片段回答问题。
规则：
1.首先找出最相关的段落，然后提炼出 1-3 个核心论点。
2.仅使用提供的文献内容作答，不得使用外部知识。
3.如果文献中没有明确答案，直接说“文献中未提及”。
文献：
{context}
问题：{question}
回答："""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
# 7. RAG 链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)
# 8. 提问循环
print("\n=== 文献 RAG（DeepSeek API）已就绪 ===")
while True:
    q = input("\n📖 你的问题：")
    if q.lower() in ["exit", "quit", "退出"]:
        break
    result = qa.invoke({"query": q})
    print("\n📝 回答：\n", result["result"])
    print("\n🔍 参考来源：")
    for i, doc in enumerate(result["source_documents"], 1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.replace("\n", " ")[:100]
        print(f"  [{i}] {os.path.basename(src)} (第{page}页): {snippet}...")