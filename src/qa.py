"""RAG 检索问答核心模块。

对应改造前的 ``RAG_base.py``（103 行顶层脚本），此处拆分为可导入、
可测试、可被 API 层复用的函数与类。

核心改进：
    1. **不再用 RetrievalQA**：该链在 LangChain 1.x 中已归入 langchain-classic
       （legacy 包）。本模块直接用 LCEL 组装检索→提示→模型→解析，
       依赖更少、可读性更强，也便于在链条中插入自定义处理；
    2. **答案带出处编号**：上下文按 ``[片段N]`` 编号，提示词要求模型标注引用，
       返回结构化 ``Answer`` 对象供 CLI/API 渲染溯源信息；
    3. **异常包装**：把 SDK 的英文报错翻译成可操作的中文提示；
    4. **流式输出**：``ask_stream()`` 支持打字机效果。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import Settings
from src.logger import get_logger
from src.vectorstore import IndexStats, VectorStoreManager

logger = get_logger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "qa_prompt.txt"


def base_name(path: str | Path) -> str:
    r"""取出路径的文件名，**不依赖当前操作系统的路径风格**。

    为什么不用 ``Path(path).name``：``Path`` 会按当前平台的分隔符解析。
    在 Linux 上 ``Path("E:\\\\papers\\\\论文.pdf").name`` 返回的是整个字符串，
    因为反斜杠在 POSIX 里不是分隔符。而文献的 ``source`` 元数据来自
    PDF 加载器，可能记录的是 Windows 风格的路径（例如索引在 Windows 上
    建立、程序在容器/Linux 上运行），这时 basename 就会失效，
    界面上会显示一长串完整路径。

    Args:
        path: 任意平台风格的路径字符串或 Path 对象。

    Returns:
        文件名部分；无法解析时返回原字符串。

    Example:
        >>> base_name(r"E:\\papers\\论文.pdf")
        '论文.pdf'
        >>> base_name("/home/user/papers/paper.pdf")
        'paper.pdf'
    """
    text = str(path)

    # Windows 风格：带盘符（C:\ 或 C:/）或以反斜杠分隔
    if "\\" in text or (len(text) > 1 and text[1] == ":"):
        return PureWindowsPath(text).name

    return PurePosixPath(text).name


@dataclass
class Source:
    """一条答案依据的文献出处。

    Attributes:
        index: 在本次回答中的引用编号（对应提示词里的 ``[片段N]``）。
        file: 文件名。
        page: 页码（非 PDF 为 0）。
        snippet: 内容摘要，供前端预览。
        score: 相似度距离（越小越相关；不同向量库口径不同，仅用于相对比较）。
    """

    index: int
    file: str
    page: object
    snippet: str
    score: float | None = None


@dataclass
class Answer:
    """一次问答的结构化结果。

    用 dataclass 而非裸字典作为返回值：调用方能获得字段补全与类型检查，
    也便于直接序列化为 FastAPI 的 JSON 响应。

    Attributes:
        text: 大模型生成的答案。
        sources: 答案依据的文献出处列表。
        latency_s: 端到端耗时（秒），用于统计 P95 延迟。
        retrieved: 本次检索到的块数（不等同于 sources，后者已按文件去重）。
    """

    text: str
    sources: list[Source] = field(default_factory=list)
    latency_s: float = 0.0
    retrieved: int = 0


class PaperRAG:
    """学术文献检索增强问答系统。

    相比改造前的脚本式实现，本类提供：
        - 惰性初始化：不构造实例就不会加载模型，方便单元测试；
        - 索引复用与增量同步：``ensure_index()`` 对比指纹后只嵌入新增文献；
        - 结构化返回：答案与出处分离，便于前端渲染与评测统计。

    Attributes:
        settings: 全局配置。
        manager: 向量库管理器。

    Example:
        >>> rag = PaperRAG(get_settings())
        >>> stats = rag.ensure_index()
        >>> answer = rag.ask("这篇论文的创新点是什么？")
        >>> print(answer.text)
    """

    def __init__(self, settings: Settings) -> None:
        """初始化问答系统（惰性加载模型，构造实例本身很轻量）。

        Args:
            settings: 全局配置。
        """
        self.settings = settings
        self.manager = VectorStoreManager(settings)
        self._llm: ChatOpenAI | None = None
        self._prompt: PromptTemplate | None = None

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def ensure_index(self, rebuild: bool = False) -> IndexStats:
        """确保向量索引与文献目录一致（增量同步）。

        Args:
            rebuild: 为 ``True`` 时全量重建。
                更换嵌入模型或调整 ``chunk_size`` 后必须重建，
                因为旧向量与新参数下的向量不可比。

        Returns:
            IndexStats: 同步统计。
        """
        return self.manager.sync(rebuild=rebuild)

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> Answer:
        """对文献库提问并返回带出处的答案。

        Args:
            question: 用户自然语言问题。
            top_k: 覆盖默认检索数量；``None`` 时用 ``settings.retrieve_top_k``。

        Returns:
            Answer: 含答案文本、出处列表与耗时。

        Raises:
            ValueError: 问题为空。
            RuntimeError: 大模型调用失败（错误信息含排查建议）。
        """
        question = self._validate_question(question)
        started = time.perf_counter()

        docs = self._retrieve(question, top_k)
        context = format_context(docs)
        prompt_text = self._ensure_prompt().format(context=context, question=question)

        try:
            response = self._ensure_llm().invoke(prompt_text)
        except Exception as exc:
            raise self._wrap_llm_error(exc) from exc

        answer = Answer(
            text=str(response.content).strip(),
            sources=collect_sources(docs),
            latency_s=round(time.perf_counter() - started, 2),
            retrieved=len(docs),
        )
        logger.info(
            "问答完成：耗时 %.2fs，检索 %d 块，引用 %d 处文献",
            answer.latency_s,
            answer.retrieved,
            len(answer.sources),
        )
        return answer

    def ask_stream(self, question: str, top_k: int | None = None) -> Iterator[str]:
        """流式问答，逐段产出答案文本（用于打字机效果）。

        Args:
            question: 用户问题。
            top_k: 覆盖默认检索数量。

        Yields:
            答案的文本增量片段。

        Raises:
            ValueError: 问题为空。
            RuntimeError: 大模型调用失败。

        Example:
            >>> for piece in rag.ask_stream("创新点是什么"):
            ...     print(piece, end="", flush=True)
        """
        question = self._validate_question(question)
        docs = self._retrieve(question, top_k)
        prompt_text = self._ensure_prompt().format(context=format_context(docs), question=question)

        try:
            for chunk in self._ensure_llm().stream(prompt_text):
                content = chunk.content
                if content:
                    yield str(content)
        except Exception as exc:
            raise self._wrap_llm_error(exc) from exc

    def retrieve(self, question: str, top_k: int | None = None) -> list[Document]:
        """只做检索不做生成，便于评测检索质量（Recall@K）。

        Args:
            question: 用户问题。
            top_k: 覆盖默认检索数量。

        Returns:
            检索到的文档块列表。
        """
        return self._retrieve(self._validate_question(question), top_k)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_question(question: str) -> str:
        """校验并规范化问题文本。

        Raises:
            ValueError: 问题为空或过长。
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > 2000:
            raise ValueError(
                f"问题过长（{len(question)} 字符）。RAG 适合具体问题，"
                "请拆分为多个小问题分别提问。"
            )
        return question

    def _retrieve(self, question: str, top_k: int | None) -> list[Document]:
        """执行检索，返回文档块列表。"""
        retriever = self.manager.as_retriever(top_k=top_k)
        docs = retriever.invoke(question)
        logger.debug("检索到 %d 个块（top_k=%s）", len(docs), top_k or self.settings.retrieve_top_k)
        return docs

    def _ensure_llm(self) -> ChatOpenAI:
        """惰性构造大模型客户端。

        注意参数名：``langchain_openai`` 1.x 中 ``model`` / ``api_key`` /
        ``base_url`` 是 Pydantic 字段别名，映射到 ``model_name`` /
        ``openai_api_key`` / ``openai_api_base``。使用别名可读性更好，
        也与 DeepSeek 官方文档的示例一致。
        """
        if self._llm is None:
            self.settings.validate()
            self._llm = ChatOpenAI(
                model=self.settings.model,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                temperature=self.settings.temperature,
                max_retries=2,  # 网络抖动自动重试，避免偶发失败中断评测
                timeout=60,
            )
        return self._llm

    def _ensure_prompt(self) -> PromptTemplate:
        """从 prompts/qa_prompt.txt 加载提示词模板。

        提示词独立成文件的好处：可以 diff、可以做 A/B 对比、可以回滚。
        这是把"提示词工程"从随手调试变成可管理资产的前提。
        """
        if self._prompt is None:
            if not _PROMPT_PATH.exists():
                raise FileNotFoundError(
                    f"提示词文件不存在：{_PROMPT_PATH}\n"
                    "  修复：从版本库恢复 prompts/qa_prompt.txt。"
                )
            template = _PROMPT_PATH.read_text(encoding="utf-8")
            self._prompt = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )
        return self._prompt

    def _wrap_llm_error(self, exc: Exception) -> RuntimeError:
        """把底层 SDK 报错包装成可操作的中文提示。"""
        detail = str(exc)
        hints = ["  排查建议："]
        if "401" in detail or "Authentication" in detail or "invalid_api_key" in detail:
            hints.append("  1) API Key 无效或已过期，请到 DeepSeek 平台重新生成；")
        elif "402" in detail or "Insufficient Balance" in detail:
            hints.append("  1) 账户余额不足，请到 DeepSeek 平台充值；")
        elif "404" in detail:
            hints.append(
                f"  1) 模型名 {self.settings.model!r} 不存在，" "请核对 .env 中的 DEEPSEEK_MODEL；"
            )
        elif "timeout" in detail.lower():
            hints.append("  1) 请求超时，请检查网络或代理设置；")
        else:
            hints.append("  1) 请检查网络能否访问 " f"{self.settings.base_url}；")
        hints.append("  2) 确认 .env 中 DEEPSEEK_BASE_URL 与 DEEPSEEK_MODEL 正确；")
        hints.append(f"  原始错误：{detail[:300]}")
        return RuntimeError("大模型调用失败。\n" + "\n".join(hints))


# ----------------------------------------------------------------------
# 模块级纯函数：便于单独测试，不依赖任何实例状态
# ----------------------------------------------------------------------
def format_context(docs: list[Document]) -> str:
    r"""把检索结果拼成带编号的上下文，供提示词引用。

    Args:
        docs: 检索到的文档块。

    Returns:
        形如 ``[片段1] 出自《xxx.pdf》第3页\n内容...`` 的拼接文本。
    """
    blocks = []
    for idx, doc in enumerate(docs, 1):
        source = base_name(doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "?")
        blocks.append(f"[片段{idx}] 出自《{source}》第{page}页\n{doc.page_content}")
    return "\n\n".join(blocks)


def collect_sources(docs: list[Document]) -> list[Source]:
    """提取出处元数据并按（文件, 页码）去重。

    去重是必要的：同一页可能被切成多个块同时命中，
    若不去重，前端会出现多条完全相同的引用。

    Args:
        docs: 检索到的文档块。

    Returns:
        去重后的出处列表，按首次出现顺序排列。
    """
    seen: set[tuple[str, object]] = set()
    sources: list[Source] = []
    for doc in docs:
        file_name = base_name(doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "?")
        key = (file_name, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                index=len(sources) + 1,
                file=file_name,
                page=page,
                snippet=doc.page_content.replace("\n", " ")[:100],
            )
        )
    return sources
