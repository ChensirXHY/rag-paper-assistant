"""``src.loader`` 单元测试。

覆盖重点：
    * 中文排版清洗（空格/控制字符/换行语义）；
    * 图表碎片过滤；
    * 加载器注册表与可选依赖降级；
    * 文件指纹（增量索引的正确性基础）；
    * 目录扫描的过滤规则与错误提示。

注意：``src.loader`` 依赖 ``langchain_community``（官方已标记 sunset）。
CI 里不安装它 —— 它会把 sentence-transformers / torch 一起拖进来，
在 GitHub Actions 上要跑十几分钟。因此这里用 ``importorskip``：
缺包时显示为"跳过"并打印原因，而不是伪装成通过，也不是无谓地报红。
本机验证（dl_env）下它是完整运行的。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "langchain_community", reason="CI 不装 langchain-community（会拖入 torch），跳过"
)

from src import loader  # noqa: E402

# ----------------------------------------------------------------------
# normalize_cjk_text
# ----------------------------------------------------------------------


class TestNormalizeCjkText:
    """中文排版清洗：把 PDF 解析出的"脏文本"还原成自然中文。"""

    def test_removes_space_between_cjk_and_punctuation(self) -> None:
        """中文与中文标点之间的空格必须删除。

        实测论文 PDF 里出现 ``"数据集 、 特征"``，这类空格会让嵌入模型
        见到非自然文本，直接影响相似度计算。
        """
        assert loader.normalize_cjk_text("数据集 、 特征") == "数据集、特征"
        assert loader.normalize_cjk_text("光伏功率预测 。") == "光伏功率预测。"
        assert loader.normalize_cjk_text("结论 ： 模型有效") == "结论：模型有效"

    def test_removes_space_between_cjk_chars(self) -> None:
        """连续中文字符之间的空格必须删除（中文不用空格分词）。"""
        assert loader.normalize_cjk_text("光伏 功率 预测") == "光伏功率预测"

    def test_removes_control_chars(self) -> None:
        """控制字符（图表坐标轴/公式符号被解析成的噪声）必须清除。

        ``\\x06`` 这类字符对嵌入模型是纯噪声，实测曾有一条这样的片段
        被排到最相关位置，挤占了真正有用的内容。
        """
        assert loader.normalize_cjk_text("信噪\x06比") == "信噪比"
        assert loader.normalize_cjk_text("\x00\x01光伏\x1f功率\x7f") == "光伏功率"

    def test_keeps_tab_and_newline(self) -> None:
        """``\\t`` 属于"保留清单"，不能被当成控制字符清掉。"""
        assert loader.normalize_cjk_text("a\tb") == "a\tb"

    def test_collapses_multi_space_but_keeps_newline(self) -> None:
        """连续空格压成一个；换行符必须原样保留。

        两条规则的优先级值得注意：**中文之间的连续空格会被整段删除**
        （``_RE_SPACE_BETWEEN_CJK`` 先去掉了它），压缩规则真正生效的场景是
        "含非中文字符的行"。所以这里用 ``"a      b"`` 验证压缩，
        用中文之间的多个空格验证"删干净" —— 两者都是期望行为。

        换行必须保留：换行是 :mod:`src.splitter` 的语义边界
        （分隔符表里 ``"\\n\\n"`` 与 ``"\\n"`` 优先级最高）。一旦在这里被压掉，
        分片就只能靠句号切，段落结构信息全部丢失 ——
        这是本模块最容易被"顺手优化"掉的行为，所以专门用一条测试守住它。
        """
        assert loader.normalize_cjk_text("光伏     功率") == "光伏功率"
        assert loader.normalize_cjk_text("a      b") == "a b"
        assert loader.normalize_cjk_text("中文  mixed     空格") == "中文 mixed 空格"
        assert loader.normalize_cjk_text("第一段\n第二段") == "第一段\n第二段"
        assert loader.normalize_cjk_text("标题\n\n正文") == "标题\n\n正文"
        # 换行数量不变：多行文本仍然是多行
        assert loader.normalize_cjk_text("一\n二\n三").count("\n") == 2

    @pytest.mark.parametrize("value", ["", "plain text", "\n\n"])
    def test_edge_cases_passthrough(self, value: str) -> None:
        """空串与纯英文/纯空白应原样返回，不能抛异常。

        真实场景：扫描版 PDF 会解析出空页面，这里必须"无输入无输出"，
        否则批量导入会在一个空页面上崩掉。
        """
        assert loader.normalize_cjk_text(value) == value

    def test_preserves_semantics_of_sentence(self) -> None:
        """整句清洗后除空格/控制字符外，内容一字不差。"""
        dirty = "本文 提出 了 一种 基于 注意力 机制 的 预测 模型 ， \x06用于 光伏 功率 预测 。"
        assert (
            loader.normalize_cjk_text(dirty)
            == "本文提出了一种基于注意力机制的预测模型，用于光伏功率预测。"
        )


# ----------------------------------------------------------------------
# is_meaningful
# ----------------------------------------------------------------------


class TestIsMeaningful:
    """图表碎片过滤：决定一个页面是否值得进向量库。"""

    def test_rejects_chart_fragment(self) -> None:
        """坐标轴刻度碎片必须被拒绝。

        ``0.00.00\\n0.25\\n0.50`` 是论文配图被解析后的典型产物，
        与任何问题的语义相似度都不稳定，却会挤占检索名额。
        """
        assert loader.is_meaningful("0.00.00\n0.25\n0.50") is False
        assert loader.is_meaningful("1.0\n0.8\n0.6\n0.4") is False
        assert loader.is_meaningful("") is False

    def test_accepts_normal_chinese_sentence(self) -> None:
        """正常中文段落必须被接受（正例同样重要，否则"全判 False"也能过测试）。"""
        assert (
            loader.is_meaningful("本文提出了一种基于注意力机制的预测模型，用于短期光伏功率预测。")
            is True
        )

    def test_threshold_is_configurable(self) -> None:
        """``min_chars`` 可调，且判定基于"去掉所有空白后的字符数"。"""
        text = "光伏功率预测"  # 6 个字符
        assert loader.is_meaningful(text) is False
        assert loader.is_meaningful(text, min_chars=6) is True
        # 空白不计入有效字符数：带空格的同一句话不会因空格而"变长"
        assert loader.is_meaningful("光 伏 功 率 预 测", min_chars=7) is False
        assert loader.is_meaningful("光 伏 功 率 预 测", min_chars=6) is True


# ----------------------------------------------------------------------
# build_registry / supported_suffixes
# ----------------------------------------------------------------------


class TestRegistry:
    """加载器注册表：格式支持与可选依赖降级。"""

    def test_contains_core_suffixes(self) -> None:
        """.pdf/.txt/.md 是核心格式，任何情况下都必须存在。"""
        registry = loader.build_registry()
        assert {".pdf", ".txt", ".md"} <= set(registry)
        # 值必须是可调用的加载器类（能被实例化并 .load()）
        for suffix in (".pdf", ".txt", ".md"):
            assert callable(registry[suffix])

    def test_docx_is_conditional_on_python_docx(self) -> None:
        """.docx 的有无取决于 python-docx 是否安装。

        这是"可选依赖降级"设计的核心：没装 python-docx 时项目照常启动，
        只在日志里提示安装方法，而不是 import 阶段直接崩掉。

        断言写成"注册表与 ``_load_docx_loader()`` 的结论必须一致"：
        支持就必须真的可加载，不支持就必须真的返回 None（不能注册一个坏类）。
        这样在装了/没装 python-docx 的机器上都成立，也不会因为
        langchain_community 内部包装了 ImportError 而给出误导性结论。
        """
        registry = loader.build_registry()
        docx_loader = loader._load_docx_loader()

        if docx_loader is None:
            assert ".docx" not in registry
        else:
            assert registry.get(".docx") is docx_loader
            # 注册表里挂着的类必须能被实例化（只是构造，不真正解析文件）
            assert callable(docx_loader)

    def test_docx_absent_when_dependency_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """降级分支：``_load_docx_loader`` 返回 None 时注册表不含 .docx。"""
        monkeypatch.setattr(loader, "_load_docx_loader", lambda: None)
        registry = loader.build_registry()
        assert ".docx" not in registry
        assert {".pdf", ".txt", ".md"} <= set(registry)

    def test_docx_present_when_dependency_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """可用分支：返回加载器类时必须注册进表（新增格式只改一行）。"""

        class _FakeDocxLoader:
            def __init__(self, file_path: str) -> None:
                self.file_path = file_path

        monkeypatch.setattr(loader, "_load_docx_loader", lambda: _FakeDocxLoader)
        assert loader.build_registry()[".docx"] is _FakeDocxLoader

    def test_supported_suffixes_matches_registry(self) -> None:
        """``supported_suffixes()`` 与注册表键集合必须一致（供错误提示使用）。"""
        assert set(loader.supported_suffixes()) == set(loader.build_registry())


# ----------------------------------------------------------------------
# compute_fingerprint
# ----------------------------------------------------------------------


class TestComputeFingerprint:
    """文件指纹：增量索引"该不该重新嵌入"的唯一依据。"""

    def test_same_file_same_fingerprint(self, tmp_dir: Path) -> None:
        """同一文件连续两次调用结果必须相等。

        否则每次启动都会误判为"文件已变更"，把全部文献重新嵌入一遍，
        增量索引直接退化成全量重建。
        """
        path = tmp_dir / "论文.txt"
        path.write_text("光伏功率预测", encoding="utf-8")
        first = loader.compute_fingerprint(path)
        second = loader.compute_fingerprint(path)
        assert first == second
        assert str(first) == str(second)
        assert first.name == "论文.txt"
        assert first.size == len("光伏功率预测".encode())

    def test_mtime_change_produces_different_fingerprint(self, tmp_dir: Path) -> None:
        """改动 mtime 后指纹必须不同（改文件内容一定会改 mtime）。"""
        path = tmp_dir / "论文.txt"
        path.write_text("光伏功率预测", encoding="utf-8")
        before = loader.compute_fingerprint(path)

        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime + 10))

        after = loader.compute_fingerprint(path)
        assert before != after
        assert before.mtime != after.mtime
        assert str(before) != str(after)

    def test_size_change_produces_different_fingerprint(self, tmp_dir: Path) -> None:
        """内容长度变化同样要能区分（size 是 mtime 之外的第二重保险）。"""
        path = tmp_dir / "论文.txt"
        path.write_text("光伏功率预测", encoding="utf-8")
        before = loader.compute_fingerprint(path)

        stat = path.stat()
        path.write_text("光伏功率预测方法研究", encoding="utf-8")
        os.utime(path, (stat.st_atime, stat.st_mtime))  # 特意保持 mtime 不变

        after = loader.compute_fingerprint(path)
        assert before != after
        assert after.size > before.size


# ----------------------------------------------------------------------
# iter_source_files
# ----------------------------------------------------------------------


class TestIterSourceFiles:
    """目录扫描：过滤 Office 临时文件、隐藏文件与不支持的格式。"""

    def test_skips_temp_hidden_and_unsupported(self, tmp_papers: Path) -> None:
        """``~$`` 临时文件、隐藏文件、不支持格式都不能出现在结果里。

        真实场景：WPS/Word 打开文档会生成 ``~$xxx.docx``（0 字节），
        若被加载会污染索引；macOS 的 ``.DS_Store`` 同理。
        """
        files = loader.iter_source_files(tmp_papers, loader.build_registry())
        names = [f.name for f in files]

        assert "论文一.txt" in names
        assert "论文二.md" in names
        assert "说明.pdf" in names
        assert not [n for n in names if n.startswith("~$")]
        assert not [n for n in names if n.startswith(".")]

    def test_respects_registry(self, tmp_dir: Path) -> None:
        """注册表里没有的扩展名一律不返回（新增格式只改注册表）。"""
        (tmp_dir / "a.txt").write_text("x", encoding="utf-8")
        (tmp_dir / "b.pdf").write_text("x", encoding="utf-8")
        (tmp_dir / "c.csv").write_text("x", encoding="utf-8")
        (tmp_dir / "d.docx").write_text("x", encoding="utf-8")

        only_txt = loader.iter_source_files(tmp_dir, {".txt": object})
        assert [f.name for f in only_txt] == ["a.txt"]

    def test_recursive_and_sorted(self, tmp_dir: Path) -> None:
        """必须递归子目录，且结果有序。

        有序保证"同一批文件每次运行得到相同的 chunk_id 分配"，
        否则评测结果无法复现。
        """
        (tmp_dir / "sub").mkdir()
        (tmp_dir / "b.txt").write_text("x", encoding="utf-8")
        (tmp_dir / "a.txt").write_text("x", encoding="utf-8")
        (tmp_dir / "sub" / "c.txt").write_text("x", encoding="utf-8")

        files = loader.iter_source_files(tmp_dir, {".txt": object})
        assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]
        assert files == sorted(files)

    def test_empty_dir_returns_empty_list(self, tmp_dir: Path) -> None:
        """空目录返回空列表（是否报错由 load_documents 决定）。"""
        assert loader.iter_source_files(tmp_dir, loader.build_registry()) == []


# ----------------------------------------------------------------------
# load_file / load_documents
# ----------------------------------------------------------------------


class TestLoadFile:
    """单文件加载：清洗 + 过滤 + 元数据补全。"""

    def test_utf8_chinese_txt_is_read_correctly(self, tmp_papers: Path) -> None:
        """``_Utf8TextLoader`` 必须能正确读出 UTF-8 中文。

        用 TextLoader 默认编码（跟随系统区域设置）读中文 Windows 上的
        UTF-8 文件会乱码，所以本项目把它固定成 UTF-8 并开启编码探测。
        """
        docs = loader.load_file(tmp_papers / "论文一.txt")
        assert len(docs) == 1

        content = docs[0].page_content
        assert "本文提出了一种基于注意力机制的光伏功率预测模型" in content
        assert "\ufffd" not in content, "出现替换字符，说明编码探测失败"

        meta = docs[0].metadata
        assert meta["file_name"] == "论文一.txt"
        assert meta["file_type"] == "txt"
        assert "fingerprint" in meta and "论文一.txt@" in meta["fingerprint"]

    def test_unsupported_suffix_returns_empty_list(self, tmp_dir: Path) -> None:
        """不支持的扩展名返回空列表，且**不抛异常**。

        批量导入时一个奇怪格式的文件不应该中断整轮索引。
        """
        path = tmp_dir / "数据.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        assert loader.load_file(path) == []

    def test_corrupt_pdf_returns_empty_list(self, tmp_papers: Path) -> None:
        """损坏文件只跳过并告警，不向上抛异常（容错隔离）。"""
        assert loader.load_file(tmp_papers / "说明.pdf") == []

    def test_short_text_page_is_filtered(self, tmp_dir: Path) -> None:
        """有效内容过少的页面必须被 is_meaningful 拦下。"""
        path = tmp_dir / "碎片.txt"
        path.write_text("0.00.00\n0.25\n0.50\n", encoding="utf-8")
        assert loader.load_file(path) == []

    def test_accepts_str_and_path(self, tmp_papers: Path) -> None:
        """``path`` 参数同时接受 str 与 Path（调用方不必先转换）。"""
        assert loader.load_file(str(tmp_papers / "论文一.txt"))
        assert loader.load_file(tmp_papers / "论文一.txt")

    @pytest.mark.integration
    def test_real_pdf_end_to_end(self, staging_project: Path) -> None:
        """真实 PDF 端到端解析（集成测试，默认跳过）。

        只验证"能解析出内容且元数据齐全"，不断言具体文字 ——
        否则换一篇论文测试就红，那种测试维护成本高于收益。
        """
        papers = sorted((staging_project / "papers").glob("*.pdf"))
        if not papers:
            pytest.skip(f"{staging_project / 'papers'} 下没有 PDF")

        docs = loader.load_file(papers[0])
        assert docs, f"未能从 {papers[0].name} 解析出任何有效页"
        assert all(d.metadata["file_type"] == "pdf" for d in docs)
        assert all(d.metadata["file_name"] == papers[0].name for d in docs)


class TestLoadDocuments:
    """整目录加载：错误提示必须可操作。"""

    def test_missing_dir_raises_with_fix_hint(self, tmp_dir: Path) -> None:
        """目录不存在时抛 FileNotFoundError，且错误信息必须含"修复"。

        这是刻意的产品决策：比起沉默地返回空列表（用户以为"文献里没有内容"，
        白白浪费一次大模型调用），不如在最前面给出一条能照做的指令。
        """
        missing = tmp_dir / "不存在的目录"
        with pytest.raises(FileNotFoundError) as excinfo:
            loader.load_documents(missing)

        message = str(excinfo.value)
        assert "修复" in message
        assert str(missing) in message

    def test_dir_without_supported_files_raises(self, tmp_dir: Path) -> None:
        """目录存在但没有受支持文件时同样报错，并列出当前支持的格式。"""
        (tmp_dir / "notes.csv").write_text("a,b\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError) as excinfo:
            loader.load_documents(tmp_dir)
        assert "未找到受支持的文件" in str(excinfo.value)

    def test_loads_all_supported_files(self, tmp_papers: Path) -> None:
        """正常目录：返回所有可解析文件的内容（``~$`` 与隐藏文件除外）。"""
        docs = loader.load_documents(tmp_papers)
        names = {d.metadata["file_name"] for d in docs}
        assert "论文一.txt" in names
        assert "论文二.md" in names
        # 临时文件与隐藏文件即使有内容也不能被加载（说明.pdf 是坏的，被跳过）
        assert not [n for n in names if n.startswith("~$")]
        assert not [n for n in names if n.startswith(".")]
        assert all("fingerprint" in d.metadata for d in docs)

    def test_accepts_str_path(self, tmp_papers: Path) -> None:
        """``root`` 参数同时接受 str 与 Path。"""
        assert loader.load_documents(str(tmp_papers))

    @pytest.mark.integration
    def test_real_papers_directory(self, staging_project: Path) -> None:
        """真实文献目录整体加载（集成测试，默认跳过）。"""
        papers = staging_project / "papers"
        if not papers.is_dir():
            pytest.skip(f"未找到文献目录：{papers}")
        docs = loader.load_documents(papers)
        assert len(docs) > 50, f"仅解析出 {len(docs)} 页，疑似解析异常"
        assert len({d.metadata["file_name"] for d in docs}) >= 6


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
