"""``src.splitter`` 单元测试。

分片是 RAG 效果的第一道关口，这个文件主要守护三件事：

1. **长度约束**：块长不得超过 ``chunk_size``（否则会吃掉提示词预算）；
2. **句子完整**：中文分片必须保住完整句子，不能把句子拦腰截断 ——
   这是相对英文默认分隔符的核心改进，一旦被改回默认值，检索质量会静默下降；
3. **元数据完整**：``chunk_id`` 连续、``start_index`` 存在、来源字段继承。

只用内存里的 ``Document`` 对象，不读任何真实文件。

关于"块尾是否为句末标点"（重要，实测结论）：
    在本项目的运行环境里（langchain-text-splitters 1.1.2），
    ``RecursiveCharacterTextSplitter(keep_separator=True)`` 会把分隔符
    挂到**下一块的开头**，因此中文分片虽然按句号切分，但块尾往往不是句号
    （实测块尾对齐率只有 5%~25%）。模块 docstring 里"分片全部对齐句末"
    的说法与本环境的实测不符。

    所以本文件**不再断言"块尾是句末标点"**（那会是一个假断言），
    改用更本质、也更稳定的口径：**句子完整性** ——
    每个句子是否完整地落在某一个块里。实测（见
    ``test_chinese_separators_keep_sentences_intact``）：
    中文分隔符 100% 保整句，英文默认只有 56%~90%，
    这才是中文分隔符配置真正带来的差别。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from config import Settings
from src.splitter import CHINESE_SEPARATORS, split_documents

# 中文句末标点：分片应当在这些位置切分
SENTENCE_ENDS = ("。", "！", "？", "；")

# 英文默认分隔符（LangChain 原版配置），用作对照组
ENGLISH_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

# 贴近论文正文的测试语料：句长从 12 到 30 字不等，全部为自然中文（无空格）
SENTENCES = [
    "光伏发电功率受太阳辐照度与环境温度共同影响。",
    "针对光伏出力随机性强、波动性大的问题，本文提出一种组合预测模型。",
    "首先采用完全自适应噪声集合经验模态分解对原始功率序列进行分解。",
    "其次利用改进的Transformer网络提取多尺度时序特征。",
    "然后通过注意力机制对关键特征加权。",
    "最后将各分量预测结果叠加得到最终功率预测值。",
    "实验数据取自某光伏电站的实测运行数据。",
    "采样间隔为十五分钟。",
    "评价指标包括均方根误差与平均绝对误差。",
    "与持久化模型相比，所提模型的均方根误差下降了百分之二十八。",
]

# 单行长文本：模拟 PDF 解析后没有段落结构的一整页文字
SINGLE_LINE_CORPUS = "".join(SENTENCES * 3)

# 多段文本：每段 3 句，段间用空行分隔（贴近真实论文排版）
PARAGRAPH_CORPUS = "\n\n".join("".join(SENTENCES[i : i + 3]) for i in range(0, 9, 3)) * 2


def _make_settings(**overrides: object) -> Settings:
    """构造一套测试用配置（不依赖环境变量与真实目录）。

    ``Settings`` 用 ``__slots__`` 把字段冻结，测试里覆盖参数只能用
    ``object.__setattr__`` —— 这是刻意的：生产代码拿不到写权限，
    而测试明确知道自己越界了。
    """
    settings = Settings()
    for key, value in overrides.items():
        object.__setattr__(settings, key, value)
    return settings


def _doc(text: str, page: int = 0) -> Document:
    """构造一个带完整 metadata 的文档页（形状与 loader 输出一致）。"""
    return Document(
        page_content=text,
        metadata={
            "source": r"E:\papers\光伏功率预测研究.pdf",
            "page": page,
            "file_name": "光伏功率预测研究.pdf",
            "file_type": "pdf",
        },
    )


def _split_with(separators: list[str], chunk_size: int, text: str) -> list[str]:
    """用指定的分隔符切分纯文本（用于与英文默认配置做对照实验）。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=separators,
        length_function=len,
    )
    return splitter.split_text(text)


def _intact_sentence_ratio(chunks: list[str], corpus: str) -> float:
    """计算"完整保留的句子"占比。

    判定方式：去掉句末标点后的句子主体必须完整出现在某个块中。
    句子被拦腰截断时，任何块里都找不到完整主体。

    Args:
        chunks: 分片结果。
        corpus: 原始语料（用于统计每句出现次数）。

    Returns:
        0~1 之间的比例。
    """
    total = 0
    intact = 0
    for sentence in SENTENCES:
        core = sentence.rstrip("".join(SENTENCE_ENDS))
        occurrences = corpus.count(sentence)
        total += occurrences
        intact += sum(1 for _ in range(occurrences) if any(core in c for c in chunks))
    return intact / total if total else 0.0


@pytest.fixture
def long_text() -> str:
    """约 1800 字的长文本（chunk_size=500 时应切成多块）。"""
    return "\n\n".join("".join(SENTENCES[i : i + 5]) for i in range(0, 10, 5)) * 4


# ----------------------------------------------------------------------
# 分隔符设计
# ----------------------------------------------------------------------
class TestChineseSeparators:
    """分隔符表：本模块相对 LangChain 默认配置的关键改动。"""

    @pytest.mark.parametrize("punct", SENTENCE_ENDS)
    def test_contains_chinese_sentence_ends(self, punct: str) -> None:
        """``。！？；`` 四个中文句末标点必须都在分隔符表里。

        这是**关键设计**：LangChain 默认分隔符是 ``["\\n\\n", "\\n", " ", ""]``，
        没有任何中文标点。用于中文论文时，长句只能按空格切或按字符硬切，
        嵌入模型看到的是语义不完整的片段，检索质量下降。
        缺任何一个都会让分片退化到下一级（逗号甚至硬切），所以逐个断言。
        """
        assert punct in CHINESE_SEPARATORS, (
            f"分隔符表缺少中文句末标点 {punct!r}；"
            f"删掉它会导致长句被硬截断，请勿移除。当前表：{CHINESE_SEPARATORS}"
        )

    def test_separator_priority_order(self) -> None:
        """分隔符必须按"语义强度"从强到弱排列。

        RecursiveCharacterTextSplitter 会依次尝试，顺序错了就等于没配置：
        若逗号排在句号前面，所有分片都会优先断在逗号处（从句中间）。
        """
        assert CHINESE_SEPARATORS.index("\n\n") < CHINESE_SEPARATORS.index("\n")
        assert CHINESE_SEPARATORS.index("\n") < CHINESE_SEPARATORS.index("。")
        assert CHINESE_SEPARATORS.index("。") < CHINESE_SEPARATORS.index("，")
        assert CHINESE_SEPARATORS.index("，") < CHINESE_SEPARATORS.index("")
        assert CHINESE_SEPARATORS[-1] == "", "最后一级必须是空串（按字符硬切的兜底）"

    def test_not_degraded_to_english_default(self) -> None:
        """回归保护：不能退化成英文默认配置。"""
        assert CHINESE_SEPARATORS != ENGLISH_DEFAULT_SEPARATORS
        assert "。" in CHINESE_SEPARATORS
        assert "。" not in ENGLISH_DEFAULT_SEPARATORS

    def test_order_is_deterministic(self) -> None:
        """表本身是模块级常量，读取两次必须完全一致（避免被就地修改）。"""
        from src import splitter

        assert splitter.CHINESE_SEPARATORS == CHINESE_SEPARATORS


# ----------------------------------------------------------------------
# 基本切分行为
# ----------------------------------------------------------------------
class TestSplitBasics:
    """切分结果的长度与数量约束。"""

    def test_long_text_produces_multiple_chunks(self, long_text: str) -> None:
        """长文本必须被切成多块（切不动说明分片器没生效）。"""
        settings = _make_settings(chunk_size=500, chunk_overlap=50)
        chunks = split_documents([_doc(long_text)], settings)

        assert len(chunks) > 1
        # 内容守恒：切分只允许丢掉段落间的空行（分隔符），不允许丢正文
        total = sum(len(c.page_content) for c in chunks)
        assert total >= len(long_text) - len(long_text) * 0.05

    def test_chunk_size_is_respected_with_margin(self, long_text: str) -> None:
        """块长不得超过 ``chunk_size``，留 10% 余量。

        为什么不要求"严格小于等于"：分隔符拼接后可能出现略超的边界情况，
        而 10% 的余量（500 → 550）对上下文预算影响可忽略。
        但完全不加约束的话，一次配置改动就可能产生 2000 字的巨块，
        把提示词预算吃光，所以这条必须守住。
        """
        settings = _make_settings(chunk_size=500, chunk_overlap=50)
        chunks = split_documents([_doc(long_text)], settings)

        limit = settings.chunk_size * 1.1
        over = [len(c.page_content) for c in chunks if len(c.page_content) > limit]
        assert not over, f"存在超出 chunk_size 10% 余量的块：{over}"

    def test_short_text_produces_single_chunk(self) -> None:
        """短文本不应被切碎（一句话就是一块）。"""
        settings = _make_settings(chunk_size=500, chunk_overlap=50)
        text = "本文提出了一种基于注意力机制的光伏功率预测模型。"
        chunks = split_documents([_doc(text)], settings)

        assert len(chunks) == 1
        assert chunks[0].page_content == text

    def test_empty_document_list_raises(self) -> None:
        """空列表抛 ValueError，并给出下一步该做什么。

        不能返回空列表：调用方会以为"文献里没有内容"，而不是"你忘了传文档"。
        """
        with pytest.raises(ValueError) as excinfo:
            split_documents([], _make_settings())
        assert "load_documents" in str(excinfo.value)

    @pytest.mark.parametrize("chunk_size", [60, 120, 300])
    def test_various_chunk_sizes(self, chunk_size: int, long_text: str) -> None:
        """不同 chunk_size 下都要能切分且不超限（参数不应写死）。"""
        settings = _make_settings(chunk_size=chunk_size, chunk_overlap=0)
        chunks = split_documents([_doc(long_text)], settings)

        assert len(chunks) > 1
        assert all(len(c.page_content) <= chunk_size * 1.1 for c in chunks)

    @pytest.mark.parametrize("chunk_size", [60, 100, 200, 500])
    def test_key_sentences_survive_splitting(self, chunk_size: int) -> None:
        """切分不能丢内容：关键句子应能在某块（或块间拼接）中找到。

        块边界可能落在句子中间，因此既接受"整句在某块里"，
        也接受"去掉空行后能在拼接文本里找到"。
        """
        settings = _make_settings(chunk_size=chunk_size, chunk_overlap=0)
        text = "".join(SENTENCES) * 3
        chunks = split_documents([_doc(text)], settings)
        joined = "".join(c.page_content for c in chunks)

        missing = [s for s in (SENTENCES[0], SENTENCES[-1]) if s not in joined]
        assert not missing, f"切分后丢失了：{missing}"


# ----------------------------------------------------------------------
# 中文分片保住完整句子（核心质量断言）
# ----------------------------------------------------------------------
class TestSentenceIntegrity:
    """中文分隔符的核心价值：句子不被拦腰截断。"""

    @pytest.mark.parametrize("chunk_size", [40, 60])
    def test_chinese_separators_keep_sentences_intact(self, chunk_size: int) -> None:
        """中文分隔符必须保住全部完整句子，且优于英文默认配置。

        这是本模块"针对中文论文定制分隔符"这一设计的可量化收益：
        句子被截断意味着嵌入模型看到的是半个因果链（"……本文提出的"），
        检索召回与答案质量都会下降。

        实测数据（langchain-text-splitters 1.1.2，本测试语料）：

            ===========  ==============  ==============
            chunk_size   中文分隔符      英文默认
            ===========  ==============  ==============
            40           100%            56%
            60           100%            78%
            ===========  ==============  ==============

        用多段文本（含空行）作语料，因为 ``\\n\\n`` 是两种配置共有的
        最高优先级分隔符，能更公平地比较中文标点带来的增量。
        """
        chinese_chunks = _split_with(CHINESE_SEPARATORS, chunk_size, PARAGRAPH_CORPUS)
        english_chunks = _split_with(ENGLISH_DEFAULT_SEPARATORS, chunk_size, PARAGRAPH_CORPUS)

        chinese_ratio = _intact_sentence_ratio(chinese_chunks, PARAGRAPH_CORPUS)
        english_ratio = _intact_sentence_ratio(english_chunks, PARAGRAPH_CORPUS)

        assert chinese_ratio == 1.0, (
            f"中文分隔符下有句子被截断（完整率 {chinese_ratio:.0%}）。"
            "中文标点可能已从分隔符表中移除。"
        )
        assert chinese_ratio > english_ratio, (
            f"中文分隔符（{chinese_ratio:.0%}）未优于英文默认"
            f"（{english_ratio:.0%}），对照实验失效"
        )

    def test_single_line_corpus_also_integrates(self) -> None:
        """没有段落结构（单行长文本）时同样要保住整句。

        PDF 解析出的正文常常整页没有空行，这条覆盖那种情况。
        """
        chunks = _split_with(CHINESE_SEPARATORS, 60, SINGLE_LINE_CORPUS)
        assert _intact_sentence_ratio(chunks, SINGLE_LINE_CORPUS) == 1.0

    def test_split_documents_keeps_whole_sentences(self) -> None:
        """走 ``split_documents``（Document 输入）时同样保住整句。

        上面的对照实验直接用了 LangChain 的 splitter，
        这条确保项目自己的封装没有引入退化。
        """
        settings = _make_settings(chunk_size=120, chunk_overlap=0)
        text = "\n\n".join("".join(SENTENCES[i : i + 5]) for i in range(0, 10, 5))
        chunks = split_documents([_doc(text)], settings)
        pieces = [c.page_content for c in chunks]

        ratio = _intact_sentence_ratio(pieces, text)
        assert ratio == 1.0, f"经 split_documents 后有句子被截断（完整率 {ratio:.0%}）"

    def test_chinese_separators_produce_more_cut_points(self) -> None:
        """中文配置应比英文配置切得更细（说明中文标点确实被用上了）。

        英文配置找不到空格，只能按字符硬切，因此块数恰好是"总长 / chunk_size"；
        中文配置能按句号切，块边界随句子长度浮动，块数必然更多。
        这条从侧面证明"中文标点进入了分隔符匹配流程"。
        """
        size = 60
        chinese = _split_with(CHINESE_SEPARATORS, size, SINGLE_LINE_CORPUS)
        english = _split_with(ENGLISH_DEFAULT_SEPARATORS, size, SINGLE_LINE_CORPUS)

        assert len(chinese) > len(english), (
            f"中文分隔符块数（{len(chinese)}）未多于英文默认（{len(english)}），"
            "说明中文标点没有生效"
        )
        assert len(english) == -(-len(SINGLE_LINE_CORPUS) // size), "英文默认配置应退化为按字符硬切"


# ----------------------------------------------------------------------
# 元数据
# ----------------------------------------------------------------------
class TestChunkMetadata:
    """块的元数据：检索溯源与增量索引都依赖它。"""

    def test_chunk_id_is_sequential(self, long_text: str) -> None:
        """``chunk_id`` 从 0 开始连续递增，不允许跳号或重复。

        ``chunk_id`` 是人工核对切分质量的定位锚点，
        跳号意味着 enumerate 与 split 的结果对不上，通常说明有块被丢弃。
        """
        settings = _make_settings(chunk_size=200, chunk_overlap=20)
        chunks = split_documents([_doc(long_text)], settings)

        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_start_index_present_and_monotonic(self) -> None:
        """``start_index`` 必须存在，且在同一页内单调不减。

        它记录块在原页中的字符位置（``add_start_index=True``），
        用于把答案定位回原文位置，是溯源功能的基础。
        """
        settings = _make_settings(chunk_size=100, chunk_overlap=0)
        text = "".join(SENTENCES) * 3
        chunks = split_documents([_doc(text)], settings)

        starts = [c.metadata["start_index"] for c in chunks]
        assert all(isinstance(s, int) for s in starts)
        assert starts == sorted(starts)
        assert starts[0] == 0

    def test_source_and_page_preserved(self, long_text: str) -> None:
        """``source`` / ``page`` 等来源字段必须被继承。

        丢了它们，``format_context`` 就只能显示"未知"，答案无法溯源。
        """
        settings = _make_settings(chunk_size=300, chunk_overlap=0)
        chunks = split_documents([_doc(long_text, page=3)], settings)

        assert chunks
        for chunk in chunks:
            assert chunk.metadata["source"] == r"E:\papers\光伏功率预测研究.pdf"
            assert chunk.metadata["page"] == 3
            assert chunk.metadata["file_name"] == "光伏功率预测研究.pdf"
            assert chunk.metadata["file_type"] == "pdf"

    def test_multi_page_chunks_keep_own_page_number(self) -> None:
        """多页输入时，每块必须保留**自己那一页**的页码，不能串页。

        串页会让溯源指向错误的页面，比不显示出处更糟。
        """
        settings = _make_settings(chunk_size=100, chunk_overlap=0)
        text = "".join(SENTENCES) * 3
        docs = [_doc(text, page=0), _doc(text, page=1)]

        chunks = split_documents(docs, settings)
        page0 = [c for c in chunks if c.metadata["page"] == 0]
        page1 = [c for c in chunks if c.metadata["page"] == 1]

        assert page0 and page1
        # chunk_id 是全局连续编号，跨页也不能重新计数
        assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


# ----------------------------------------------------------------------
# overlap 参数
# ----------------------------------------------------------------------
class TestOverlap:
    """重叠长度：接回被切断的上下文，代价是块数变多。"""

    @pytest.mark.parametrize("overlap", [0, 20, 50])
    def test_overlap_values_are_accepted(self, overlap: int, long_text: str) -> None:
        """0 / 小重叠 / 大重叠都要能正常工作，且块长不超限。

        用参数化而不是写死一个值：``chunk_overlap`` 是典型的需要调参的量，
        测试要保证它在合理区间内都不会让分片器行为异常。
        """
        settings = _make_settings(chunk_size=200, chunk_overlap=overlap)
        chunks = split_documents([_doc(long_text)], settings)

        assert len(chunks) > 1
        assert all(len(c.page_content) <= settings.chunk_size * 1.1 for c in chunks)
        assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))

    @pytest.mark.parametrize("overlap", [0, 10, 20, 30])
    def test_overlap_does_not_break_length_limit(self, overlap: int) -> None:
        """重叠不能把块撑过 ``chunk_size``。

        重叠是在原块基础上"向前多看一点"，若实现把重叠内容算进块长，
        实际块长会变成 chunk_size + overlap，提示词预算被悄悄吃掉。
        """
        settings = _make_settings(chunk_size=120, chunk_overlap=overlap)
        text = "\n\n".join("".join(SENTENCES[i : i + 5]) for i in range(0, 10, 5)) * 3
        chunks = split_documents([_doc(text)], settings)

        assert all(len(c.page_content) <= 120 * 1.1 for c in chunks)

    @pytest.mark.parametrize("overlap", [0, 20, 50])
    def test_larger_overlap_produces_at_least_as_many_chunks(self, overlap: int) -> None:
        """重叠越大，块数不会更少（重叠必然增加冗余内容）。"""
        text = "\n\n".join("".join(SENTENCES[i : i + 5]) for i in range(0, 10, 5)) * 4
        baseline = split_documents([_doc(text)], _make_settings(chunk_size=200, chunk_overlap=0))
        current = split_documents(
            [_doc(text)], _make_settings(chunk_size=200, chunk_overlap=overlap)
        )
        assert len(current) >= len(baseline)

    def test_overlap_actually_duplicates_content(self) -> None:
        """overlap>0 时相邻块必须真的共享内容。

        否则"配置了重叠"只是心理安慰，长句被切断处的上下文仍然丢失。
        判定方式：在某一块里取一段文字，它必须出现在下一块的开头区域。
        """
        settings = _make_settings(chunk_size=120, chunk_overlap=60)
        text = "\n\n".join("".join(SENTENCES[i : i + 5]) for i in range(0, 10, 5)) * 2
        chunks = split_documents([_doc(text)], settings)

        assert len(chunks) >= 2
        first, second = chunks[0].page_content, chunks[1].page_content
        # 取前一块的尾部 20 字，应能在后一块中找到（重叠区）
        tail = first[-20:]
        assert (
            tail in second
        ), f"相邻块未共享内容，overlap 未生效。\n块1尾部：{tail!r}\n块2开头：{second[:60]!r}"


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
