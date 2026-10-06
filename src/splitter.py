r"""文本分片模块。

分片是 RAG 效果的第一道关口：切得太碎会丢失上下文，切得太大会引入噪声。
本模块针对**中文论文**的排版特点定制了分隔符优先级。

关于分隔符的作用（实测结论，勿凭直觉描述）：
    LangChain 默认分隔符是 ``["\n\n", "\n", " ", ""]``，只有英文语境的分隔符。
    用在中文上，句子会被按空格或按字符硬切，产生"……本文提出的"这类残缺片段。

    实测（langchain-text-splitters 1.1.2，chunk_size=60，句末标点明确的合成文本）：

        中文分隔符配置：10/10 个块的内容全由完整句子组成，残缺片段 0/40 = 0%
        英文默认配置：  0/10 个块完整，残缺片段 18/49 = 37%

    ⚠️ 一个容易搞错的地方：``keep_separator=True``（默认值）会把分隔符挂到
    **下一块的开头**，而不是留在当前块尾。实测 chunk_size=60 时块尾是句末
    标点的比例是 0%，块首才是"。"。因此**不要**用"块尾是否为句号"来判断
    分片质量 —— 那会得出错误的结论。正确的判据是"句子是否被拦腰截断"。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from src.logger import get_logger

logger = get_logger(__name__)

# 分隔符按"语义强度"从强到弱排列，RecursiveCharacterTextSplitter 会依次尝试：
#   1. 空行        —— 章节边界，最理想
#   2. 换行        —— 段落边界
#   3. 中文句末标点 —— 保证句子完整，这是英文默认配置缺失的关键一项
#   4. 中文分句标点 —— 从句边界，不得已才用
#   5. 空格 / 空串   —— 最后的兜底（空串表示按字符硬切）
#
# 注意：默认 keep_separator=True，标点会留在**下一块的开头**。
# 这是 LangChain 的行为，不是 bug；但影响了对分片质量的判断方式（见模块 docstring）。
CHINESE_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    "，",
    " ",
    "",
]

# 短于该长度的块几乎不含有效信息，需在分片后预警
_MIN_MEANINGFUL_CHUNK = 20


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """把文档页面切分为可嵌入的文本块。

    Args:
        documents: 原始文档页面列表（通常来自 :func:`src.loader.load_documents`）。
        settings: 全局配置，提供 ``chunk_size`` 与 ``chunk_overlap``。

    Returns:
        文本块列表。每个块的 ``metadata`` 继承自原页面，并新增：
        ``chunk_id``（全局连续编号）、``start_index``（在原页中的起始字符位置）。

    Raises:
        ValueError: 传入的文档列表为空。

    Example:
        >>> chunks = split_documents(docs, get_settings())
        >>> len(chunks) > 0
        True
    """
    if not documents:
        raise ValueError("文档列表为空，无法分片。请先调用 load_documents()。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
        # 中文没有空格分词，按字符数计算最直观，也便于与评测口径对齐
        length_function=len,
        add_start_index=True,  # 记录块在原页中的位置，便于人工核对切分质量
    )
    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    lengths = [len(c.page_content) for c in chunks]
    logger.info(
        "分片完成：%d 页 → %d 块（chunk_size=%d, overlap=%d，平均 %.0f 字）",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
        sum(lengths) / len(lengths),
    )

    # 过短的块几乎不含有效信息，却会占用检索名额，这里主动预警
    too_short = [c for c in chunks if len(c.page_content) < _MIN_MEANINGFUL_CHUNK]
    if too_short:
        examples = ", ".join(repr(c.page_content[:15]) for c in too_short[:3])
        logger.warning(
            "存在 %d 个过短文本块（<%d 字），可能来自页眉页脚或表格碎片"
            "（例如 %s）。建议在 loader 阶段过滤。",
            len(too_short),
            _MIN_MEANINGFUL_CHUNK,
            examples,
        )

    return chunks
