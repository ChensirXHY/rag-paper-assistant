"""``src.qa`` 单元测试。

这个文件只测**纯函数与数据类**，一行网络请求都不发：

    * :func:`format_context`：上下文编号 → 提示词里 ``[片段N]`` 的引用基础；
    * :func:`collect_sources`：出处去重 → 前端不出现重复引用；
    * :meth:`PaperRAG._validate_question`：入参守卫 → 空问题不会白花一次调用；
    * ``Answer`` / ``Source`` 数据类：返回值契约。

为什么不测 ``ask()``：它必然要调大模型（需要有效 API Key、联网、花钱、
结果还不确定）。那类端到端行为由 ``eval/run_eval.py`` 在真实数据上度量，
单元测试里只保证"喂给模型的上下文拼装正确" —— 拼错了模型再强也答不对。

``importorskip`` 说明（两道保护，都不是"掩盖失败"）：
    * ``src.qa`` 顶层 import ``langchain_openai``（大模型客户端）；
    * ``src.qa`` 还通过 ``src.vectorstore`` 间接依赖
      ``langchain_community``（Chroma 与 BGE 集成包）。
    两者都属于"部署时按需安装"的集成包，缺任意一个都会在**收集阶段**
    直接 ImportError（不是测试失败）。用 ``importorskip`` 显式跳过并打印
    原因，比让整套测试红掉更有信息量 —— 但这也意味着：CI 必须装上它们，
    否则这些用例不会被执行（取舍见 .github/workflows/ci.yml）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_openai", reason="缺 langchain-openai（大模型客户端）")
pytest.importorskip(
    "langchain_community", reason="缺 langchain-community（src.vectorstore 依赖它）"
)

from langchain_core.documents import Document  # noqa: E402

from src.qa import Answer, PaperRAG, Source, collect_sources, format_context  # noqa: E402


def _doc(
    content: str,
    source: str = r"E:\papers\光伏功率预测研究.pdf",
    page: object = 0,
) -> Document:
    """构造一个检索结果块（metadata 形状与 loader 输出一致）。"""
    return Document(page_content=content, metadata={"source": source, "page": page})


# ----------------------------------------------------------------------
# format_context
# ----------------------------------------------------------------------
class TestFormatContext:
    """把检索结果拼成带编号的上下文。"""

    def test_numbers_and_file_names(self) -> None:
        """输出必须含 ``[片段1]`` ``[片段2]`` 编号与文件名。

        编号是引用机制的锚点：提示词要求模型写"依据[片段1]"，
        少了编号，答案就无法溯源到具体文献。
        """
        docs = [
            _doc("TCN 用于提取局部时序特征。", source=r"E:\papers\论文一.pdf", page=0),
            _doc("BiLSTM 用于捕捉长程依赖。", source=r"E:\papers\论文二.pdf", page=1),
        ]
        context = format_context(docs)

        assert "[片段1]" in context
        assert "[片段2]" in context
        assert "论文一.pdf" in context
        assert "论文二.pdf" in context
        assert "TCN 用于提取局部时序特征。" in context
        assert "BiLSTM 用于捕捉长程依赖。" in context

    def test_page_number_is_included(self) -> None:
        """页码必须写进上下文，供答案标注"第几页"。"""
        context = format_context([_doc("正文", page=7)])
        assert "第7页" in context

    def test_two_chunks_are_separated_by_blank_line(self) -> None:
        """块之间用空行分隔，便于模型区分不同片段。"""
        context = format_context([_doc("甲"), _doc("乙")])
        assert "\n\n" in context

    def test_empty_input_returns_empty_string(self) -> None:
        """没有检索结果时返回空串，不能抛异常。

        ``ask()`` 会先拼上下文再调模型，检索空结果时应该让模型回答
        "文献中未提及"，而不是在拼串阶段崩掉。
        """
        assert format_context([]) == ""

    def test_missing_metadata_is_tolerated(self) -> None:
        """``source`` 缺失时显示"未知"，不能抛 KeyError。"""
        context = format_context([Document(page_content="内容", metadata={})])
        assert "[片段1]" in context
        assert "未知" in context
        assert "内容" in context

    def test_only_base_name_is_shown(self) -> None:
        """只显示文件名而非完整路径（路径又长又没有信息量）。"""
        context = format_context([_doc("内容", source=r"E:\very\long\path\论文三.pdf")])
        assert "论文三.pdf" in context
        assert r"E:\very\long\path" not in context

    def test_numbering_follows_input_order(self) -> None:
        """编号必须与输入顺序一致（检索结果的排序就是相关性排序）。"""
        docs = [_doc(f"第{i}块内容") for i in range(1, 5)]
        context = format_context(docs)
        positions = [context.index(f"[片段{i}]") for i in range(1, 5)]
        assert positions == sorted(positions)


# ----------------------------------------------------------------------
# collect_sources
# ----------------------------------------------------------------------
class TestCollectSources:
    """出处提取与去重。"""

    def test_dedup_by_file_and_page(self) -> None:
        """按（文件, 页码）去重：同一页被切成多块同时命中时只留一条。

        不去重的话，前端会出现多条一模一样的引用，用户会以为
        "有四个不同来源支持这个结论"，实际只有一个。
        """
        docs = [
            _doc("内容A", source=r"E:\papers\论文一.pdf", page=0),
            _doc("内容B", source=r"E:\papers\论文一.pdf", page=0),
            _doc("内容C", source=r"E:\papers\论文一.pdf", page=1),
            _doc("内容D", source=r"E:\papers\论文二.pdf", page=0),
        ]
        sources = collect_sources(docs)

        assert len(sources) == 3
        assert [(s.file, s.page) for s in sources] == [
            ("论文一.pdf", 0),
            ("论文一.pdf", 1),
            ("论文二.pdf", 0),
        ]

    def test_index_starts_at_one_and_is_continuous(self) -> None:
        """``index`` 从 1 开始连续递增（对应提示词里的 ``[片段N]``）。"""
        docs = [
            _doc("A", source=r"E:\p\一.pdf", page=0),
            _doc("B", source=r"E:\p\二.pdf", page=2),
            _doc("C", source=r"E:\p\三.pdf", page=3),
        ]
        sources = collect_sources(docs)

        assert [s.index for s in sources] == [1, 2, 3]

    def test_first_occurrence_order_is_kept(self) -> None:
        """去重后按**首次出现顺序**排列（不能重排，否则引用编号会跳）。"""
        docs = [
            _doc("A", source=r"E:\p\z.pdf", page=0),
            _doc("B", source=r"E:\p\a.pdf", page=0),
            _doc("C", source=r"E:\p\z.pdf", page=0),
        ]
        assert [s.file for s in collect_sources(docs)] == ["z.pdf", "a.pdf"]

    def test_snippet_is_truncated_and_flattened(self) -> None:
        """摘要要压掉换行并截断（前端表格一行显示，不能撑破布局）。"""
        docs = [_doc("第一行\n第二行\n" + "很长的内容" * 40)]
        snippet = collect_sources(docs)[0].snippet

        assert "\n" not in snippet
        assert len(snippet) <= 100

    def test_score_defaults_to_none(self) -> None:
        """当前实现不填 ``score``（LangChain 的检索器不返回距离），默认 None。"""
        assert collect_sources([_doc("A")])[0].score is None

    def test_empty_input(self) -> None:
        """无检索结果时返回空列表。"""
        assert collect_sources([]) == []

    def test_multiple_pages_of_same_file_are_distinct(self) -> None:
        """同一文件的不同页是不同出处（不能只按文件去重）。

        否则"答案来自第 3 页和第 7 页"会被压成一条，丢掉一半溯源信息。
        """
        docs = [
            _doc("A", source=r"E:\p\论文.pdf", page=3),
            _doc("B", source=r"E:\p\论文.pdf", page=7),
        ]
        assert [(s.file, s.page) for s in collect_sources(docs)] == [
            ("论文.pdf", 3),
            ("论文.pdf", 7),
        ]

    def test_page_none_and_zero_are_distinct(self) -> None:
        """``page`` 缺失（"?"）与非 PDF 的 0 是两种不同情况，都要保留。"""
        docs = [
            Document(page_content="A", metadata={"source": r"E:\p\a.txt"}),
            Document(page_content="B", metadata={"source": r"E:\p\a.txt", "page": 0}),
        ]
        sources = collect_sources(docs)
        assert len(sources) == 2
        assert sources[0].page == "?"
        assert sources[1].page == 0


# ----------------------------------------------------------------------
# PaperRAG._validate_question
# ----------------------------------------------------------------------
class TestValidateQuestion:
    """入参守卫：在花掉一次大模型调用之前拦下无效问题。"""

    @pytest.mark.parametrize("question", ["", "   ", "\t\n  ", "\n", None])
    def test_empty_question_raises(self, question) -> None:
        """空串/纯空格/None 都要抛 ValueError。

        ``None`` 也要覆盖：API 层收到缺字段的 JSON 时传进来的就是 None，
        这里必须报"问题不能为空"，而不是在下游某处抛 AttributeError。
        """
        with pytest.raises(ValueError, match="问题不能为空"):
            PaperRAG._validate_question(question)

    def test_too_long_question_raises(self) -> None:
        """超过 2000 字符抛 ValueError，并提示拆分问题。

        上限的意义：RAG 适合具体问题。把整段论文贴进来当问题，
        检索只会命中与开头几个词相似的块，答案必然是错的 ——
        与其给出错误答案，不如明确要求拆分。
        """
        with pytest.raises(ValueError) as excinfo:
            PaperRAG._validate_question("光" * 2001)
        message = str(excinfo.value)
        assert "过长" in message
        assert "2001" in message

    def test_exactly_2000_chars_is_accepted(self) -> None:
        """边界值 2000 必须放行（上限是"超过才拒绝"）。"""
        assert len(PaperRAG._validate_question("光" * 2000)) == 2000

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  论文的创新点是什么？  ", "论文的创新点是什么？"),
            ("\n用了什么评价指标\t", "用了什么评价指标"),
            (
                "TCN-BiLSTM-Attention-ESN 的创新点是什么？",
                "TCN-BiLSTM-Attention-ESN 的创新点是什么？",
            ),
        ],
    )
    def test_normal_question_is_stripped(self, raw: str, expected: str) -> None:
        """正常问题返回 strip 后的值。

        必须 strip：前后空格会让嵌入向量与库里的文本对不齐（细微但真实），
        也会污染评测报告里的问题列表。
        """
        assert PaperRAG._validate_question(raw) == expected

    def test_is_static_and_pure(self) -> None:
        """校验逻辑是静态方法，不需要构造实例（因此不需要 API Key）。"""
        assert isinstance(PaperRAG.__dict__["_validate_question"], staticmethod)
        # 不构造 PaperRAG 也能调用，说明它与实例状态无关
        assert PaperRAG._validate_question("正常问题") == "正常问题"

    def test_does_not_mutate_semantics(self) -> None:
        """只去首尾空白，不动中间内容（包括全角空格以外的中间空格）。"""
        raw = "本文 与 那篇 的 区别 是 什么"
        assert PaperRAG._validate_question(raw) == raw


# ----------------------------------------------------------------------
# 数据类契约
# ----------------------------------------------------------------------
class TestDataclasses:
    """返回值契约：字段与默认值一旦变化，CLI/API/评测脚本都会受影响。"""

    def test_answer_defaults(self) -> None:
        """``Answer`` 只有 ``text`` 必填，其余都有安全默认值。"""
        answer = Answer(text="模型回答")

        assert answer.text == "模型回答"
        assert answer.sources == []
        assert answer.latency_s == 0.0
        assert answer.retrieved == 0

    def test_answer_accepts_full_payload(self) -> None:
        """完整字段可写入（``ask()`` 的实际用法）。"""
        source = Source(index=1, file="论文一.pdf", page=3, snippet="摘要")
        answer = Answer(text="回答", sources=[source], latency_s=1.25, retrieved=4)

        assert answer.sources[0] is source
        assert answer.latency_s == 1.25
        assert answer.retrieved == 4

    def test_answer_sources_are_not_shared(self) -> None:
        """两个实例的 ``sources`` 必须互相独立（可变默认值必须用 factory）。"""
        first, second = Answer(text="A"), Answer(text="B")
        first.sources.append(Source(index=1, file="x.pdf", page=0, snippet="s"))
        assert second.sources == []

    def test_source_required_fields_and_defaults(self) -> None:
        """``Source`` 必填 4 项，``score`` 默认 None。"""
        source = Source(index=2, file="论文二.pdf", page=1, snippet="片段")

        assert source.index == 2
        assert source.file == "论文二.pdf"
        assert source.page == 1
        assert source.snippet == "片段"
        assert source.score is None

    def test_source_page_accepts_object(self) -> None:
        """``page`` 声明为 ``object``：既可能是 int，也可能是 "?"（非 PDF）。

        真实数据里两种情况都存在，类型标注放宽是有意为之。
        """
        assert Source(index=1, file="a.txt", page="?", snippet="").page == "?"
        assert Source(index=1, file="a.pdf", page=0, snippet="").page == 0

    def test_source_score_can_be_set(self) -> None:
        """保留 ``score`` 字段以便将来接入带距离的检索器（如 similarity_search_with_score）。"""
        assert Source(index=1, file="a.pdf", page=0, snippet="", score=0.42).score == 0.42


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
