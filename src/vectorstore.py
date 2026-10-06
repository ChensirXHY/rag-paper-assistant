"""向量库封装模块：构建、增量更新、查询。

本模块修复了原始实现中一个会**静默降低检索质量**的缺陷：

    原代码用 ``if not os.path.exists(CHROMA_DB_DIR)`` 判断是否重建索引。
    只要目录存在就跳过重建 —— 于是往 papers/ 里新增文献后，
    新文献永远不会被嵌入，检索也永远查不到它们，而且没有任何报错。

    实测本项目：papers/ 有 6 篇文献、共 446 个文本块，
    但向量库里只有 152 条（仅前 2 篇），另外 4 篇从未被索引。

本模块改用**文件指纹比对**做增量索引：
    已索引文件的指纹集合与磁盘现状对比，只嵌入新增/变更的文件，
    并删除已从磁盘移除文件的残留向量。既修掉漏索引，又不必每次全量重建。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import Settings
from src.loader import build_registry, compute_fingerprint, iter_source_files, load_file
from src.logger import get_logger
from src.splitter import split_documents

logger = get_logger(__name__)

# 集合名固定，便于识别；换知识库时改环境变量或此常量
_COLLECTION_NAME = "paper_knowledge_base"

# 余弦距离：BGE 系列模型用归一化向量 + 余弦相似度，这是官方推荐配置。
# 若用默认的 l2 距离，归一化向量的排序结果虽然接近但不严格等价。
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


@dataclass
class IndexStats:
    """一次索引同步的结果统计。

    Attributes:
        total_files: 磁盘上的文献文件总数。
        indexed_files: 同步后库中实际覆盖的文件数。
        total_chunks: 同步后库中的文本块总数。
        added_chunks: 本次新增的块数。
        removed_files: 本次清理的、已从磁盘删除的文件。
        unchanged_files: 指纹未变、被跳过的文件。
    """

    total_files: int = 0
    indexed_files: int = 0
    total_chunks: int = 0
    added_chunks: int = 0
    removed_files: list[str] = field(default_factory=list)
    unchanged_files: list[str] = field(default_factory=list)

    @property
    def is_up_to_date(self) -> bool:
        """索引是否与磁盘现状一致（无新增、无删除）。"""
        return self.added_chunks == 0 and not self.removed_files


class VectorStoreManager:
    """Chroma 向量库管理器，负责索引同步与检索器构造。

    Attributes:
        settings: 全局配置。

    Example:
        >>> manager = VectorStoreManager(get_settings())
        >>> stats = manager.sync()          # 增量同步索引
        >>> retriever = manager.as_retriever()
    """

    def __init__(self, settings: Settings) -> None:
        """初始化管理器（不加载任何模型，模型在首次访问属性时才加载）。

        Args:
            settings: 全局配置。
        """
        self.settings = settings
        self._embeddings: HuggingFaceBgeEmbeddings | None = None
        self._store: Chroma | None = None

    # ------------------------------------------------------------------
    # 惰性资源
    # ------------------------------------------------------------------
    @property
    def embeddings(self) -> HuggingFaceBgeEmbeddings:
        """嵌入模型（首次访问时加载，约需数秒）。

        ``normalize_embeddings=True`` 是 BGE 系列模型的官方要求：
        BGE 用余弦相似度检索，不归一化会让长文本因向量模长偏大而虚高相似度。
        """
        if self._embeddings is None:
            device = self.settings.resolve_device()
            logger.info("加载嵌入模型 %s（device=%s）...", self.settings.embedding_model, device)
            self._embeddings = HuggingFaceBgeEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def store(self) -> Chroma:
        """向量库实例（首次访问时连接/创建）。"""
        if self._store is None:
            self.settings.persist_dir.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.settings.persist_dir),
                collection_metadata=_COLLECTION_METADATA,
            )
            logger.debug("已连接向量库 %s", self.settings.persist_dir)
        return self._store

    # ------------------------------------------------------------------
    # 索引同步
    # ------------------------------------------------------------------
    def _indexed_fingerprints(self) -> dict[str, set[str]]:
        """读取库中各文件已索引的指纹集合。

        Returns:
            形如 ``{文件名: {指纹字符串, ...}}`` 的字典。
            一个文件可能有多条指纹（文件被改动过多次且未清理旧向量），
            因此用集合而非单值。
        """
        data = self.store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        result: dict[str, set[str]] = {}
        for meta in metadatas:
            if not meta:
                continue
            name = str(meta.get("file_name", ""))
            fingerprint = str(meta.get("fingerprint", ""))
            if name:
                result.setdefault(name, set()).add(fingerprint)
        return result

    def _delete_file(self, file_name: str) -> None:
        """删除某文件在库中的全部向量。

        Args:
            file_name: 文件名（与 metadata 中的 ``file_name`` 对应）。
        """
        try:
            self.store.delete(where={"file_name": file_name})
        except Exception as exc:
            # 删除失败不致命：旧向量最多让检索多召回一点过时内容
            logger.warning("清理 %s 的旧向量失败：%s", file_name, exc)

    def sync(self, rebuild: bool = False) -> IndexStats:
        """把向量库与磁盘上的文献目录同步（增量）。

        Args:
            rebuild: 为 ``True`` 时先清空整个集合再全量重建。
                适用于更换嵌入模型或调整分片参数后（此时旧向量已失效）。

        Returns:
            IndexStats: 本次同步的统计结果。

        Raises:
            FileNotFoundError: 文献目录为空或无受支持文件。
            ValueError: 配置校验失败（如分片参数非法）。
        """
        self.settings.validate(require_api_key=False)

        if rebuild:
            logger.warning("rebuild=True，删除现有集合后全量重建")
            try:
                self.store.delete_collection()
            except Exception as exc:
                # 集合可能本就不存在，这不算错误
                logger.debug("删除集合时遇到可忽略的问题：%s", exc)
            # delete_collection 后需要重新创建，否则后续 add_documents 会失败
            self._store = None

        stats = IndexStats()
        registry_files = iter_source_files(self.settings.papers_dir, build_registry())
        stats.total_files = len(registry_files)
        disk_fingerprints = {p.name: str(compute_fingerprint(p)) for p in registry_files}

        indexed = self._indexed_fingerprints()

        # --- 1. 清理已从磁盘删除的文件 ---
        for name in set(indexed) - set(disk_fingerprints):
            logger.info("文献已从磁盘移除，清理其向量：%s", name)
            self._delete_file(name)
            stats.removed_files.append(name)

        # --- 2. 找出需要（重新）索引的文件 ---
        to_index = []
        for path in registry_files:
            name = path.name
            expected = disk_fingerprints[name]
            if expected in indexed.get(name, set()):
                stats.unchanged_files.append(name)
                continue
            if name in indexed:
                # 文件被改动过：先删旧向量再重新嵌入，避免新旧内容同时被检索到
                logger.info("文献内容已变更，重建其向量：%s", name)
                self._delete_file(name)
            to_index.append(path)

        # --- 3. 增量嵌入 ---
        if to_index:
            logger.info("需索引 %d 个文献：%s", len(to_index), ", ".join(p.name for p in to_index))
            for path in to_index:
                added = self._index_single_file(path)
                stats.added_chunks += added
        else:
            logger.info("所有 %d 个文献均已是最新，跳过嵌入", len(registry_files))

        # --- 4. 汇总 ---
        final_indexed = self._indexed_fingerprints()
        stats.indexed_files = len(final_indexed)
        stats.total_chunks = self.store._collection.count()

        if stats.is_up_to_date:
            logger.info(
                "索引已是最新：%d 个文件、%d 个文本块", stats.indexed_files, stats.total_chunks
            )
        else:
            logger.info(
                "同步完成：新增 %d 块，清理 %d 个文件；现有 %d 个文件、%d 个文本块",
                stats.added_chunks,
                len(stats.removed_files),
                stats.indexed_files,
                stats.total_chunks,
            )

        # 一致性自检：库里覆盖的文件数应等于磁盘文件数
        if stats.indexed_files != stats.total_files:
            logger.warning(
                "索引覆盖不完整：磁盘 %d 个文件，库中仅 %d 个。" "请检查上方是否有解析失败的文件。",
                stats.total_files,
                stats.indexed_files,
            )

        return stats

    def _index_single_file(self, path) -> int:
        """加载并嵌入单个文件，返回新增的块数。

        Args:
            path: 文献文件路径。

        Returns:
            该文件产生的文本块数量；解析失败返回 0。
        """
        pages = load_file(path)
        if not pages:
            logger.warning("文件中没有可索引的文本页：%s", path.name)
            return 0

        chunks = split_documents(pages, self.settings)
        self.store.add_documents(chunks)
        logger.info("已索引 %s：%d 页 → %d 块", path.name, len(pages), len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def as_retriever(
        self,
        top_k: int | None = None,
        search_type: str = "mmr",
    ) -> VectorStoreRetriever:
        """构造检索器。

        Args:
            top_k: 返回的文档块数量；``None`` 时用 ``settings.retrieve_top_k``。
            search_type: ``"mmr"`` 或 ``"similarity"``。

        Returns:
            可直接 ``invoke(question)`` 的检索器。

        Note:
            默认用 MMR（最大边际相关性）而非纯相似度检索：
            纯 top-k 会让同一段落的相邻分片挤占全部检索名额，
            导致"创新点是什么"这类多点问题只能召回到同一处内容。
            MMR 在相关性与多样性之间折中，``mmr_lambda=0.5`` 表示两者等权。
        """
        k = top_k or self.settings.retrieve_top_k

        if search_type == "mmr":
            search_kwargs = {
                "k": k,
                # fetch_k 是重排前的候选池。取 k 的数倍才有"多样性"可选择空间；
                # 若 fetch_k == k，MMR 退化为普通相似度检索。
                "fetch_k": max(self.settings.fetch_k, k * 4),
                "lambda_mult": self.settings.mmr_lambda,
            }
        else:
            search_kwargs = {"k": k}

        return self.store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def collection_info(self) -> dict[str, object]:
        """返回向量库的概况，便于诊断与展示。

        Returns:
            含集合名、块数、覆盖文件数的字典。
        """
        indexed = self._indexed_fingerprints()
        return {
            "集合名": _COLLECTION_NAME,
            "文本块数": self.store._collection.count(),
            "覆盖文件数": len(indexed),
            "文件列表": sorted(indexed.keys()),
        }
