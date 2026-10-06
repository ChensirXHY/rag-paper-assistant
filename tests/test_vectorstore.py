"""``src.vectorstore`` 单元测试（全程 mock，不加载真实嵌入模型）。

这个文件守护的是本模块**修掉的那个静默缺陷**：

    原实现用 ``if not os.path.exists(CHROMA_DB_DIR)`` 判断是否重建索引 ——
    只要目录存在就跳过重建，于是新增文献永远不会被嵌入，检索也永远查不到，
    而且没有任何报错。实测 6 篇文献只有前 2 篇进了库。

现在改用文件指纹比对做增量索引，所以测试的重点就是"增量的四种判断分支"：

    ==================  ==========================  ====================
    库中状态             磁盘状态                     期望行为
    ==================  ==========================  ====================
    空                   3 个文件                     全部索引
    有匹配指纹           文件未变                     跳过（unchanged）
    指纹不匹配           文件被改                     先 delete 再 add
    有残留指纹           文件已删除                   出现在 removed_files
    ==================  ==========================  ====================

mock 策略：
    * ``HuggingFaceBgeEmbeddings``：真加载要 1.3 GB 模型 + 数十秒，测试里
      用 ``PropertyMock`` 直接替换 ``manager.embeddings``；
    * ``Chroma``：用一个小号假 store（记录调用、维护 metadata 列表），
      它准确复现了真实 Chroma 的三个关键接口：``get(include=...)``、
      ``delete(where=...)``、``_collection.count()``。
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

pytest.importorskip(
    "langchain_community", reason="CI 不装 langchain-community（会拖入 torch），跳过"
)

from langchain_core.documents import Document  # noqa: E402

from src.vectorstore import IndexStats, VectorStoreManager  # noqa: E402


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------
class FakeStore:
    """最小可用的 Chroma 替身。

    只实现 :class:`VectorStoreManager` 真正用到的接口。它维护一份
    ``_metadatas`` 列表而不是真做向量检索 —— 本项目所有增量判断逻辑
    都只看 metadata 里的 ``file_name`` 与 ``fingerprint``，
    因此这份替身足以覆盖全部分支。

    Attributes:
        _metadatas: 库中所有块的 metadata（模拟 Chroma 的存储）。
        add_documents_calls: 每次 add 收到的块列表，供断言。
        delete_calls: 每次 delete 的 where 条件，供断言。
        deleted_collections: ``delete_collection`` 被调用的次数。
    """

    def __init__(self, metadatas: list[dict] | None = None) -> None:
        self._metadatas: list[dict] = list(metadatas or [])
        self.add_documents_calls: list[list[Document]] = []
        self.delete_calls: list[dict] = []
        self.deleted_collections = 0
        self.collection = MagicMock()
        self.collection.count.side_effect = lambda: len(self._metadatas)
        self._collection = self.collection

    # --- 真实 Chroma 的接口子集 ---
    def get(self, include: list[str] | None = None) -> dict:
        """返回库中全部 metadata（``include=["metadatas"]``）。"""
        return {"metadatas": [dict(m) for m in self._metadatas]}

    def delete(self, where: dict) -> None:
        """按 metadata 过滤删除，语义与 Chroma 一致。"""
        self.delete_calls.append(where)
        name = where.get("file_name")
        self._metadatas = [m for m in self._metadatas if m.get("file_name") != name]

    def delete_collection(self) -> None:
        """清空整个集合。"""
        self.deleted_collections += 1
        self._metadatas = []

    def add_documents(self, documents: list[Document]) -> None:
        """写入块：把 metadata 追加进库，模拟真实落库效果。"""
        self.add_documents_calls.append(list(documents))
        self._metadatas.extend(dict(d.metadata) for d in documents)

    def as_retriever(self, search_type: str, search_kwargs: dict):
        """返回一个哨兵对象，便于断言参数传递是否正确。"""
        self.last_retriever_args = {"search_type": search_type, "search_kwargs": search_kwargs}
        return ("retriever", search_type, search_kwargs)


# ----------------------------------------------------------------------
# Fixture
# ----------------------------------------------------------------------
@pytest.fixture
def make_manager(settings, monkeypatch: pytest.MonkeyPatch):
    """构造 VectorStoreManager 工厂：嵌入模型与向量库全部用替身。

    返回一个 ``make(metadatas=None) -> (manager, store)`` 函数：
    每个测试可以按需要预置"库中已有内容"，从而精确命中某个增量分支。

    Args:
        settings: conftest 提供的真实 Settings（目录都已指向 tmp_dir）。
        monkeypatch: 用于替换 ``load_file``（避免真去解析文件内容）。

    Returns:
        callable: 工厂函数。
    """
    from src import vectorstore as vs

    def _fake_load_file(path):
        """用文件名造一页假文档，形状与 loader.load_file 的真实输出一致。

        关键点：必须带上 ``fingerprint`` 与 ``file_name`` ——
        ``VectorStoreManager`` 的增量判断完全依赖这两个字段
        （真实 loader 会写入它们，见 ``src/loader.py`` 的
        ``page.metadata["fingerprint"] = str(fingerprint)``）。
        少了 fingerprint，第二轮同步会被误判成"文件已变更"。
        """
        from src.loader import compute_fingerprint

        return [
            Document(
                page_content=f"{path.name} 的正文内容，用于测试增量索引逻辑。" * 3,
                metadata={
                    "source": str(path),
                    "page": 0,
                    "file_name": path.name,
                    "fingerprint": str(compute_fingerprint(path)),
                },
            )
        ]

    monkeypatch.setattr(vs, "load_file", _fake_load_file)

    def make(metadatas: list[dict] | None = None):
        manager = VectorStoreManager(settings)
        store = FakeStore(metadatas)
        monkeypatch.setattr(type(manager), "store", PropertyMock(return_value=store), raising=True)
        monkeypatch.setattr(
            type(manager),
            "embeddings",
            PropertyMock(return_value=MagicMock(name="fake-embeddings")),
            raising=True,
        )
        return manager, store

    return make


@pytest.fixture
def papers_dir(tmp_dir):
    """三个假的 .txt 文献（不生成 PDF，CI 里也不需要 pypdf）。

    注意 ``exist_ok=True``：``settings`` fixture 也会准备同名目录，
    两个 fixture 谁先执行都不该报错。
    """
    papers = tmp_dir / "papers"
    papers.mkdir(exist_ok=True)
    for name in ("a.txt", "b.txt", "c.txt"):
        (papers / name).write_text("光伏功率预测方法研究。", encoding="utf-8")
    return papers


def _fingerprint_of(papers_dir, name: str) -> str:
    """计算某文件的真实指纹字符串（与 indexer 用的是同一函数）。"""
    from src.loader import compute_fingerprint

    return str(compute_fingerprint(papers_dir / name))


def _meta(name: str, fingerprint: str) -> dict:
    """构造一条库中 metadata 记录。"""
    return {"file_name": name, "fingerprint": fingerprint}


# ----------------------------------------------------------------------
# IndexStats
# ----------------------------------------------------------------------
class TestIndexStats:
    """索引同步结果：``is_up_to_date`` 是 UI 与日志的判定依据。"""

    @pytest.mark.parametrize(
        ("added", "removed", "expected"),
        [
            (0, [], True),  # 无新增无删除 → 最新
            (5, [], False),  # 有新增 → 不是最新
            (0, ["旧文献.pdf"], False),  # 有清理 → 不是最新
            (5, ["旧文献.pdf"], False),
        ],
    )
    def test_is_up_to_date(self, added: int, removed: list[str], expected: bool) -> None:
        """只有"没加也没删"才算最新。

        注意 ``unchanged_files`` 不参与判定：跳过多少文件不影响"是否最新"。
        """
        stats = IndexStats(added_chunks=added, removed_files=removed)
        assert stats.is_up_to_date is expected

    def test_defaults(self) -> None:
        """默认值必须是"空统计"，否则新建对象会误报有增量。"""
        stats = IndexStats()
        assert stats.total_files == 0
        assert stats.indexed_files == 0
        assert stats.total_chunks == 0
        assert stats.added_chunks == 0
        assert stats.removed_files == []
        assert stats.unchanged_files == []
        assert stats.is_up_to_date is True

    def test_mutable_default_is_not_shared(self) -> None:
        """两个实例的列表字段必须互不影响（避免 dataclass 的可变默认值坑）。"""
        first, second = IndexStats(), IndexStats()
        first.removed_files.append("x.pdf")
        assert second.removed_files == []


# ----------------------------------------------------------------------
# sync：四种增量分支
# ----------------------------------------------------------------------
class TestSyncEmptyLibrary:
    """库为空：所有文件都要被索引（对应"修掉了漏索引"这一条）。"""

    def test_all_files_indexed(self, make_manager, papers_dir) -> None:
        manager, store = make_manager(metadatas=[])
        stats = manager.sync()

        assert stats.total_files == 3
        assert stats.added_chunks == 3  # 每个假文件产出 1 块
        assert stats.indexed_files == 3
        assert stats.total_chunks == 3
        assert stats.unchanged_files == []
        assert stats.removed_files == []
        assert len(store.add_documents_calls) == 3
        assert store.delete_calls == []
        assert stats.is_up_to_date is False

    def test_indexed_metadatas_carry_fingerprint(self, make_manager, papers_dir) -> None:
        """写入的块必须带 ``file_name`` 与 ``fingerprint``。

        少了 fingerprint，下次同步就无法判断"是否需要重新索引"，
        整个增量机制失效（等于退回"永远全量重建"）。
        """
        manager, store = make_manager(metadatas=[])
        manager.sync()

        for meta in store._metadatas:
            assert meta["file_name"].endswith(".txt")
            assert "@" in meta["fingerprint"]

    def test_empty_papers_dir_syncs_to_zero(self, make_manager, settings) -> None:
        """文献目录为空时同步是空操作，不抛异常。

        ``settings`` fixture 造的就是一个空 papers 目录。真实场景：
        用户新装了项目还没放文献，``python app.py`` 不应该崩，
        只报"库中 0 个文件、0 个文本块"即可。
        """
        manager, store = make_manager(metadatas=[])
        stats = manager.sync()

        assert stats.total_files == 0
        assert stats.added_chunks == 0
        assert stats.indexed_files == 0
        assert stats.total_chunks == 0
        assert store.add_documents_calls == []

    def test_missing_papers_dir_raises(self, make_manager, settings) -> None:
        """文献目录**不存在**时抛 FileNotFoundError，且提示里带修复方法。

        与上一条的区别很关键：目录不存在是配置错误（必须报错），
        目录为空只是还没放数据（不该报错）。
        """
        settings.papers_dir.rmdir()  # 删掉空目录
        manager, _ = make_manager(metadatas=[])
        with pytest.raises(FileNotFoundError) as excinfo:
            manager.sync()
        assert "修复" in str(excinfo.value)


class TestSyncUnchanged:
    """指纹一致：跳过嵌入，这是增量索引省时间的来源。"""

    def test_unchanged_files_are_skipped(self, make_manager, papers_dir) -> None:
        metadatas = [
            _meta(name, _fingerprint_of(papers_dir, name)) for name in ("a.txt", "b.txt", "c.txt")
        ]
        manager, store = make_manager(metadatas=metadatas)

        stats = manager.sync()

        assert sorted(stats.unchanged_files) == ["a.txt", "b.txt", "c.txt"]
        assert stats.added_chunks == 0
        assert stats.indexed_files == 3
        assert stats.total_chunks == 3
        assert store.add_documents_calls == [], "指纹未变却仍在嵌入，增量索引失效"
        assert store.delete_calls == []
        assert stats.is_up_to_date is True

    def test_partially_unchanged(self, make_manager, papers_dir) -> None:
        """部分文件未变、部分文件是新的：只索引新的那部分。"""
        metadatas = [_meta("a.txt", _fingerprint_of(papers_dir, "a.txt"))]
        manager, store = make_manager(metadatas=metadatas)

        stats = manager.sync()

        assert stats.unchanged_files == ["a.txt"]
        assert stats.added_chunks == 2
        assert len(store.add_documents_calls) == 2
        assert stats.indexed_files == 3


class TestSyncChanged:
    """指纹不一致：必须先删旧向量再写新向量，否则新旧内容同时被检索到。"""

    def test_changed_file_is_deleted_then_readded(self, make_manager, papers_dir) -> None:
        stale = "a.txt@1:100"  # 与真实指纹必然不同，模拟"文件被改过"
        metadatas = [_meta("a.txt", stale)]
        manager, store = make_manager(metadatas=metadatas)

        stats = manager.sync()

        # 1) 先删旧向量：where 条件精确指向该文件
        assert store.delete_calls == [{"file_name": "a.txt"}]
        # 2) 再重新嵌入（a 被改，b/c 是新增）
        assert stats.added_chunks == 3
        assert stats.unchanged_files == []
        # 3) 库里该文件只剩新指纹，旧指纹已被清掉
        fingerprints = {m["fingerprint"] for m in store._metadatas if m["file_name"] == "a.txt"}
        assert fingerprints == {_fingerprint_of(papers_dir, "a.txt")}
        assert stale not in fingerprints

    def test_rebuild_clears_collection_first(self, make_manager, papers_dir) -> None:
        """``rebuild=True`` 时必须先删集合，再全量重建。

        换嵌入模型或改 chunk_size 后（向量空间已变），旧向量必须作废。
        """
        metadatas = [_meta("a.txt", _fingerprint_of(papers_dir, "a.txt"))]
        manager, store = make_manager(metadatas=metadatas)

        stats = manager.sync(rebuild=True)

        assert store.deleted_collections == 1
        assert stats.added_chunks == 3  # 即使指纹一致也全部重嵌
        assert len(store.add_documents_calls) == 3

    def test_rebuild_tolerates_missing_collection(self, make_manager, papers_dir) -> None:
        """集合本就不存在时，``rebuild=True`` 不能崩（删集合失败可忽略）。"""
        manager, store = make_manager(metadatas=[])
        store.delete_collection = MagicMock(side_effect=ValueError("Collection not found"))

        stats = manager.sync(rebuild=True)
        assert stats.added_chunks == 3


class TestSyncRemoved:
    """磁盘上被删除的文件：其残留向量必须清理，否则检索会返回幽灵内容。"""

    def test_removed_file_is_cleaned(self, make_manager, papers_dir) -> None:
        # 库里有 4 个文件，磁盘上只有 3 个（手动删掉一个）
        (papers_dir / "d.txt").write_text("将被删除的文献。", encoding="utf-8")
        metadatas = [
            _meta(name, _fingerprint_of(papers_dir, name))
            for name in ("a.txt", "b.txt", "c.txt", "d.txt")
        ]
        (papers_dir / "d.txt").unlink()

        manager, store = make_manager(metadatas=metadatas)
        stats = manager.sync()

        assert stats.removed_files == ["d.txt"]
        assert store.delete_calls == [{"file_name": "d.txt"}]
        assert stats.added_chunks == 0
        assert stats.unchanged_files == ["a.txt", "b.txt", "c.txt"]
        assert stats.total_files == 3
        assert stats.indexed_files == 3
        assert stats.is_up_to_date is False
        assert all(m["file_name"] != "d.txt" for m in store._metadatas)

    def test_delete_failure_is_not_fatal(
        self, make_manager, papers_dir, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """清理旧向量失败只能告警，不能让整个同步中断。

        设计取舍：留下一条过时向量最多让检索多召回一点旧内容，
        而中断同步会让用户完全用不了系统 —— 两害相权取其轻。
        """
        metadatas = [_meta("a.txt", "a.txt@1:100")]
        manager, store = make_manager(metadatas=metadatas)
        store.delete = MagicMock(side_effect=RuntimeError("collection unavailable"))

        stats = manager.sync()
        assert stats.added_chunks == 3  # 仍然完成了索引


# ----------------------------------------------------------------------
# _indexed_fingerprints
# ----------------------------------------------------------------------
class TestIndexedFingerprints:
    """读取库中已索引指纹：增量判断的输入。"""

    def test_groups_by_file_name(self, make_manager) -> None:
        """同一文件的多条指纹要合并成集合（历史上被改过多次的情况）。"""
        manager, _ = make_manager(
            metadatas=[
                {"file_name": "a.pdf", "fingerprint": "a.pdf@1:1"},
                {"file_name": "a.pdf", "fingerprint": "a.pdf@2:2"},
                {"file_name": "b.pdf", "fingerprint": "b.pdf@3:3"},
            ]
        )
        result = manager._indexed_fingerprints()
        assert result == {"a.pdf": {"a.pdf@1:1", "a.pdf@2:2"}, "b.pdf": {"b.pdf@3:3"}}

    def test_ignores_incomplete_metadata(self, make_manager) -> None:
        """缺 ``file_name`` 或空 metadata 的记录要跳过，不能抛异常。

        真实库里可能混入早期版本写入的、缺字段的向量。
        """
        manager, _ = make_manager(metadatas=[{}, {"fingerprint": "x"}, {"file_name": "c.pdf"}])
        assert manager._indexed_fingerprints() == {"c.pdf": {""}}


# ----------------------------------------------------------------------
# as_retriever
# ----------------------------------------------------------------------
class TestAsRetriever:
    """检索器参数：MMR 的 ``fetch_k`` 是"多样性"能否生效的关键。"""

    def test_mmr_search_kwargs(self, make_manager) -> None:
        """MMR 模式必须带 k / fetch_k / lambda_mult 三个参数。"""
        manager, store = make_manager()
        manager.as_retriever()

        args = store.last_retriever_args
        assert args["search_type"] == "mmr"
        assert set(args["search_kwargs"]) == {"k", "fetch_k", "lambda_mult"}
        assert args["search_kwargs"]["k"] == manager.settings.retrieve_top_k
        assert args["search_kwargs"]["lambda_mult"] == manager.settings.mmr_lambda

    @pytest.mark.parametrize("top_k", [1, 4, 10])
    def test_mmr_fetch_k_at_least_four_times_k(self, make_manager, top_k: int) -> None:
        """``fetch_k >= k * 4``：候选池太小时 MMR 会退化成普通相似度检索。

        ``fetch_k == k`` 时"从 k 个里挑 k 个"没有任何选择空间，
        MMR 的意义（让多点问题召回不同段落）就消失了。
        """
        manager, store = make_manager()
        manager.as_retriever(top_k=top_k)

        kwargs = store.last_retriever_args["search_kwargs"]
        assert kwargs["k"] == top_k
        assert kwargs["fetch_k"] >= top_k * 4

    def test_similarity_search_kwargs(self, make_manager) -> None:
        """``similarity`` 模式只传 k：多传 fetch_k 会被 LangChain 报错。"""
        manager, store = make_manager()
        manager.as_retriever(top_k=3, search_type="similarity")

        args = store.last_retriever_args
        assert args["search_type"] == "similarity"
        assert args["search_kwargs"] == {"k": 3}

    def test_top_k_defaults_to_settings(self, make_manager) -> None:
        """不传 top_k 时用配置里的 ``retrieve_top_k``（不要在代码里写死 4）。"""
        manager, store = make_manager()
        manager.as_retriever()
        assert store.last_retriever_args["search_kwargs"]["k"] == manager.settings.retrieve_top_k

    def test_returns_store_result(self, make_manager) -> None:
        """``as_retriever`` 应原样返回 store 造好的检索器（不额外包装）。"""
        manager, store = make_manager()
        result = manager.as_retriever()
        assert result == ("retriever", "mmr", store.last_retriever_args["search_kwargs"])


# ----------------------------------------------------------------------
# collection_info
# ----------------------------------------------------------------------
class TestCollectionInfo:
    """向量库概况：诊断"文献没被索引"这类问题的入口。"""

    def test_reports_blocks_and_files(self, make_manager) -> None:
        manager, _ = make_manager(
            metadatas=[
                {"file_name": "a.pdf", "fingerprint": "a.pdf@1:1"},
                {"file_name": "a.pdf", "fingerprint": "a.pdf@1:1"},
                {"file_name": "b.pdf", "fingerprint": "b.pdf@2:2"},
            ]
        )
        info = manager.collection_info()

        assert info["文本块数"] == 3
        assert info["覆盖文件数"] == 2
        assert info["文件列表"] == ["a.pdf", "b.pdf"]

    def test_empty_library(self, make_manager) -> None:
        """空库不能报错（首次运行时的正常状态）。"""
        manager, _ = make_manager(metadatas=[])
        info = manager.collection_info()
        assert info["文本块数"] == 0
        assert info["覆盖文件数"] == 0
        assert info["文件列表"] == []


# ----------------------------------------------------------------------
# 集成测试（默认跳过）
# ----------------------------------------------------------------------
@pytest.mark.integration
class TestVectorStoreIntegration:
    """用真实 Chroma + 临时目录验证同步结果（不加载真实嵌入模型）。"""

    def test_sync_with_real_chroma(
        self, settings, papers_dir, tmp_dir, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """真实 Chroma（临时目录）跑两轮同步，验证增量判断在真实库上同样成立。

        嵌入函数用确定性替身：这里要验证的是"Chroma 接口调用是否正确"
        （``get(include=['metadatas'])`` / ``delete(where=...)`` / ``count()``），
        而不是嵌入质量 —— 后者属于 ``eval/run_eval.py`` 的职责。
        """
        pytest.importorskip("chromadb", reason="未安装 chromadb")
        import hashlib

        from langchain_community.vectorstores import Chroma
        from langchain_core.embeddings import Embeddings

        from src import vectorstore as vs

        class _HashEmbeddings(Embeddings):
            """把文本哈希成固定维度向量，仅用于让 Chroma 能落库。"""

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._vec(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._vec(text)

            @staticmethod
            def _vec(text: str) -> list[float]:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                return [b / 255.0 for b in digest[:16]]

        # 复用与单元测试相同的假 load_file，保证只验证向量库这一层。
        # 注意 fingerprint 必须写进 metadata —— 增量判断完全依赖它；
        # 漏掉它会让"第二轮同步应全部跳过"断言失败，而问题其实出在测试替身上。
        def _fake_load_file(path):
            from src.loader import compute_fingerprint

            return [
                Document(
                    page_content="真实 Chroma 增量同步测试正文。" * 3,
                    metadata={
                        "source": str(path),
                        "page": 0,
                        "file_name": path.name,
                        "fingerprint": str(compute_fingerprint(path)),
                    },
                )
            ]

        monkeypatch.setattr(vs, "load_file", _fake_load_file)

        embeddings = _HashEmbeddings()
        manager = VectorStoreManager(settings)
        manager._embeddings = embeddings
        manager._store = Chroma(
            collection_name="test_collection",
            embedding_function=embeddings,
            persist_directory=str(tmp_dir / "chroma"),
        )

        stats = manager.sync()
        assert stats.added_chunks == 3
        assert stats.indexed_files == 3
        assert stats.total_chunks == 3

        # 第二轮：指纹一致，必须全部跳过（真实的 get(include=...) 路径）
        stats2 = manager.sync()
        assert stats2.added_chunks == 0
        assert sorted(stats2.unchanged_files) == ["a.txt", "b.txt", "c.txt"]
        assert stats2.is_up_to_date is True

        # 删掉一个磁盘文件后，其残留向量必须被清理
        (papers_dir / "c.txt").unlink()
        stats3 = manager.sync()
        assert stats3.removed_files == ["c.txt"]
        assert stats3.total_chunks == 2
