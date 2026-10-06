"""文档加载模块：统一处理 PDF / Word / Markdown 多格式文献。

对应简历中「PDF/Word 多格式文献批量解析」的能力描述 ——
改造前的实现只用了 ``DirectoryLoader + PyPDFLoader``，实际只支持 PDF。

设计要点：
    1. **注册表模式**：新增格式只需在 ``_build_registry`` 加一行，不改主流程；
    2. **可选依赖降级**：python-docx 未安装时，Word 支持自动禁用并给出安装提示，
       而不是让整个程序 import 失败 —— 这让项目在最小依赖下也能跑起来；
    3. **容错隔离**：单个文件损坏只跳过该文件并告警，不中断批量导入；
    4. **文件指纹**：为每个文件计算大小+mtime，供增量索引用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# 文本清洗规则
# ----------------------------------------------------------------------
# PDF 解析器常在中日韩字符与标点之间插入空格，实测本项目论文里出现
# "proposed ， including"、"数据集 、 特征" 这类形式。
# 这类空格会：
#   1. 让嵌入向量偏离原始语义（模型见到的是非自然文本）；
#   2. 在展示给用户的出处摘要里显得很脏。
# 中文排版本身不用空格分隔标点，因此可以安全移除这类空格。
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_CJK_PUNCT = "\u3000-\u303f\uff00-\uffef"

_RE_SPACE_BEFORE_PUNCT = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK_PUNCT}])")
_RE_SPACE_AFTER_PUNCT = re.compile(rf"(?<=[{_CJK_PUNCT}])[ \t]+(?=[{_CJK}])")
_RE_SPACE_BETWEEN_CJK = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
# 同一行内的多个空格压成一个（保留换行，因为换行是分片的语义边界）
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# PDF 里的图表坐标轴、公式符号常被解析成一串控制字符，
# 例如 "\x06\n\x04 \x04 \x07 \x05 \r \x06"。这些字符对嵌入模型是纯噪声，
# 却会占用检索名额（实测曾有一条这样的片段被排到最相关位置）。
# 这里的做法是：保留 \n \t \r，其余 C0/C1 控制字符一律清除。
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 非空白字符中至少这么多个才算有效内容
MIN_MEANINGFUL_CHARS = 30


def normalize_cjk_text(text: str) -> str:
    r"""清洗 PDF 解析产生的中文排版噪声。

    处理项：
        1. 清除控制字符（图表轴标签、公式符号常被解析成 ``\x06`` 这类噪声）；
        2. 移除中文字符与中文标点之间的多余空格；
        3. 移除中文字符之间的多余空格；
        4. 压掉行内连续空格。

    注意**不会**动换行符：换行是 :mod:`src.splitter` 的语义边界，
    压掉会让段落信息丢失，反而降低分片质量。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。

    Example:
        >>> normalize_cjk_text("数据集 、 特征 工程")
        '数据集、特征工程'
        >>> normalize_cjk_text("信噪\x06比")
        '信噪比'
    """
    if not text:
        return text
    text = _RE_CONTROL_CHARS.sub("", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub("", text)
    text = _RE_SPACE_AFTER_PUNCT.sub("", text)
    text = _RE_SPACE_BETWEEN_CJK.sub("", text)
    return _RE_MULTI_SPACE.sub(" ", text)


def is_meaningful(text: str, min_chars: int = MIN_MEANINGFUL_CHARS) -> bool:
    r"""判断一段文本是否值得进入向量库。

    过滤掉"看起来有内容、实际是图表碎片"的页。实测论文 PDF 中，
    图片里的坐标轴刻度会被解析成 ``0.00.00\n0.25\n0.50`` 这类文本，
    它们与任何问题的语义相似度都不稳定，却会挤占检索名额。

    Args:
        text: 待判定文本。
        min_chars: 有效字符数下限。

    Returns:
        ``True`` 表示内容可用。

    Example:
        >>> is_meaningful("0.00.00\n0.25\n0.50")
        False
        >>> is_meaningful("本文提出了一种基于注意力机制的预测模型。")
        True
    """
    stripped = re.sub(r"\s", "", text)
    return len(stripped) >= min_chars


@dataclass(frozen=True)
class FileFingerprint:
    """文件指纹，用于判断文献是否需要重新索引。

    只用 ``size + mtime`` 而非完整哈希：对几十 MB 的 PDF 逐字节哈希
    每次启动都要多花数秒，而 size+mtime 在实践中已足够判别文件是否变更。
    """

    name: str
    size: int
    mtime: float

    def __str__(self) -> str:
        """返回 ``文件名@大小:修改时间`` 形式的指纹字符串（写入向量元数据用）。"""
        return f"{self.name}@{self.size}:{self.mtime:.0f}"


def compute_fingerprint(path: Path) -> FileFingerprint:
    """计算单个文件的指纹。

    Args:
        path: 文件路径。

    Returns:
        含文件名、字节数、修改时间的指纹对象。
    """
    stat = path.stat()
    return FileFingerprint(name=path.name, size=stat.st_size, mtime=stat.st_mtime)


def _load_docx_loader() -> type | None:
    """按需导入 Word 加载器。

    ``Docx2txtLoader`` 依赖 ``python-docx``，而该包不一定安装。
    这里做成可选导入：安装了就用，没装就降级并提示，
    避免"因为一个可选格式没装，整个系统起不来"。

    Returns:
        Docx2txtLoader 类；不可用时返回 ``None``。
    """
    try:
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader
    except Exception as exc:  # ImportError 或其底层依赖缺失
        logger.warning(
            "Word 解析不可用（%s）。如需支持 .docx，请执行：pip install python-docx",
            type(exc).__name__,
        )
        return None


def build_registry() -> dict[str, type]:
    """构建「扩展名 → 加载器」注册表。

    用注册表代替 if/elif 链：新增格式只需追加一行，
    主流程 ``load_documents`` 无需改动（开闭原则）。

    Returns:
        形如 ``{".pdf": PyPDFLoader, ...}`` 的字典。
    """
    registry: dict[str, type] = {
        ".pdf": PyPDFLoader,
        # TextLoader 默认 encoding=None 会依赖系统区域设置，
        # 中文 Windows 上读取 UTF-8 文件会乱码，因此这里用偏函数固定编码。
        ".txt": _Utf8TextLoader,
        ".md": _Utf8TextLoader,
    }
    docx_loader = _load_docx_loader()
    if docx_loader is not None:
        registry[".docx"] = docx_loader
    return registry


class _Utf8TextLoader(TextLoader):
    """强制以 UTF-8 读取的文本加载器（自动探测编码作为兜底）。"""

    def __init__(self, file_path: str) -> None:
        """初始化加载器。

        Args:
            file_path: 待读取的文本文件路径。
        """
        super().__init__(file_path, encoding="utf-8", autodetect_encoding=True)


def supported_suffixes() -> tuple[str, ...]:
    """返回当前环境实际支持的扩展名元组。"""
    return tuple(build_registry().keys())


def iter_source_files(root: Path, registry: dict[str, type]) -> list[Path]:
    """列出目录下所有受支持的文献文件。

    Args:
        root: 文献根目录。
        registry: 扩展名注册表。

    Returns:
        排序后的文件路径列表。已排除 Office 编辑产生的 ``~$`` 临时文件
        与隐藏文件，否则会加载到一堆 0 字节垃圾。
    """
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in registry:
            continue
        # Office/WPS 打开文档时会生成 ~$xxx.docx 临时文件；隐藏文件同样跳过
        if path.name.startswith("~$") or path.name.startswith("."):
            logger.debug("跳过临时文件 %s", path.name)
            continue
        files.append(path)
    return files


def load_file(path: str | Path) -> list[Document]:
    """加载单个文献文件。

    与 :func:`load_documents` 的区别：只解析指定文件，不做目录扫描。
    增量索引时按文件调用它，避免"每索引一个文件就把整个目录重新解析一遍"
    （6 篇论文的目录下，那样会重复解析 6 次，白等好几分钟）。

    Args:
        path: 文献文件路径。

    Returns:
        该文件的文档页面列表；文件不受支持或解析失败时返回空列表。

    Example:
        >>> pages = load_file("./papers/xxx.pdf")
        >>> pages[0].metadata["file_type"]
        'pdf'
    """
    path = Path(path)
    registry = build_registry()
    loader_cls = registry.get(path.suffix.lower())
    if loader_cls is None:
        logger.warning(
            "不支持的格式 %s（当前支持：%s）",
            path.suffix,
            ", ".join(registry.keys()),
        )
        return []

    fingerprint = compute_fingerprint(path)
    try:
        pages = loader_cls(str(path)).load()
    except Exception as exc:
        logger.warning("解析失败，跳过 %s：%s", path.name, exc)
        return []

    cleaned = 0
    kept: list[Document] = []
    dropped = 0
    for page in pages:
        before = page.page_content
        page.page_content = normalize_cjk_text(before)
        if page.page_content != before:
            cleaned += 1

        # 过滤图表碎片页：这些页的文本多是坐标轴刻度或公式符号，
        # 嵌入后与任何问题的相似度都不稳定，却会挤占检索名额。
        if not is_meaningful(page.page_content):
            dropped += 1
            continue

        page.metadata["file_name"] = path.name
        page.metadata["file_type"] = path.suffix.lstrip(".").lower()
        page.metadata["fingerprint"] = str(fingerprint)
        kept.append(page)

    logger.debug(
        "已加载 %s（%d 页，%.1f MB；清洗 %d 页，过滤 %d 个低质页）",
        path.name,
        len(pages),
        fingerprint.size / 1024 / 1024,
        cleaned,
        dropped,
    )
    if dropped and not kept:
        logger.warning(
            "%s 的 %d 页全部被判为低质内容（可能是扫描版或纯图表页）",
            path.name,
            dropped,
        )
    return kept


def load_documents(root: str | Path) -> list[Document]:
    """递归加载目录下所有受支持的文献文件。

    Args:
        root: 文献所在目录，支持 ``str`` 或 ``Path``。

    Returns:
        文档页面列表。每个 Document 的 ``metadata`` 包含：
        ``source``（绝对路径）、``page``（页码，非 PDF 时为 0）、
        ``file_name``（文件名）、``file_type``（扩展名）、``fingerprint``（指纹）。

    Raises:
        FileNotFoundError: 目录不存在，或目录下没有任何受支持的文件。
            选择显式报错而非静默返回空列表，因为空列表会让调用方
            误以为"文献里没有内容"，从而白白浪费一次大模型调用。

    Example:
        >>> docs = load_documents("./papers")
        >>> docs[0].metadata["file_type"]
        'pdf'
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"文献目录不存在：{root}\n"
            "  修复：创建该目录并放入 PDF/DOCX 文件，"
            "或用环境变量 PAPERS_DIR 指定其他路径。"
        )

    registry = build_registry()
    files = iter_source_files(root, registry)
    if not files:
        raise FileNotFoundError(
            f"{root} 下未找到受支持的文件" f"（当前环境支持：{', '.join(registry.keys())}）"
        )

    documents: list[Document] = []
    skipped: list[str] = []

    # 复用 load_file 而不是重复一遍加载+清洗逻辑：
    # 单一实现路径意味着"单文件索引"与"整目录加载"的行为必然一致，
    # 不会出现两处逻辑各自演化、结果不同的经典问题。
    for path in files:
        pages = load_file(path)
        if not pages:
            skipped.append(path.name)
            continue
        documents.extend(pages)

    if not documents:
        raise FileNotFoundError(
            f"{root} 下的 {len(files)} 个文件全部解析失败：{', '.join(skipped)}\n"
            "  常见原因：PDF 是扫描件（无文字层，需先 OCR）或文件已加密。"
        )

    logger.info(
        "文献加载完成：%d 个文件、%d 页%s",
        len(files) - len(skipped),
        len(documents),
        f"，跳过 {len(skipped)} 个异常文件" if skipped else "",
    )
    return documents
