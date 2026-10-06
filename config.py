"""全局配置模块。

设计原则：
    1. 所有可变参数集中在此，业务代码不出现魔法值；
    2. 配置来源优先级：环境变量 > .env 文件 > 本文件默认值；
    3. 启动即校验，把错误暴露在最前面，而不是在程序深处抛异常。

关于 .env 路径（重要）：
    ``load_dotenv()`` 不带参数时，会**从调用它的文件所在目录向上查找** .env，
    而不是从当前工作目录查找。这意味着从别的目录启动脚本时，
    .env 可能静默加载失败（返回值 False），导致 API Key 变成 None，
    最后在调用大模型时才报一个难以定位的错误。

    本模块用 ``Path(__file__)`` 显式锚定项目根目录，无论从哪里启动都能正确加载，
    并在加载失败时直接给出可操作的提示。

用法：
    >>> from config import get_settings
    >>> settings = get_settings()
    >>> settings.chunk_size
    500
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录：本文件所在位置，所有相对路径以此为基准
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

# 显式指定路径，避免"从别的目录启动就读不到配置"这个经典坑
_ENV_LOADED = load_dotenv(ENV_FILE)


def _env_str(key: str, default: str) -> str:
    """读取字符串型环境变量，空字符串视为未设置。"""
    value = os.getenv(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    """读取整数型环境变量。

    不直接写 ``int(os.getenv(key))``，是为了避免 `CHUNK_SIZE=abc` 这类笔误
    在程序深处抛出难以定位的 ValueError，这里会明确指出是哪个变量有问题。
    """
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 应为整数，实际为 {raw!r}。请检查 .env 文件。") from exc


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，非法值给出明确报错。"""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 应为浮点数，实际为 {raw!r}。请检查 .env 文件。") from exc


class Settings:
    """项目运行配置（不可变）。

    所有字段在实例化时从环境变量读取一次并冻结，
    避免运行期被意外改写导致"行为不一致却找不到原因"。

    Attributes:
        papers_dir: 待索引的文献目录。
        persist_dir: Chroma 向量库持久化目录。
        embedding_model: 嵌入模型名称或本地路径。
        chunk_size: 文本分片长度（字符数）。
        chunk_overlap: 相邻分片的重叠长度。
        retrieve_top_k: 每次检索返回的文档块数量。
        fetch_k: MMR 重排前的候选池大小。
        mmr_lambda: MMR 相关性权重，0 为纯相似度、1 为纯多样性。
        device: 嵌入模型运行设备，auto 表示自动探测。
        log_level: 日志级别。
        api_key / base_url / model: 大模型接口配置。
    """

    __slots__ = (
        "papers_dir",
        "persist_dir",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "retrieve_top_k",
        "fetch_k",
        "mmr_lambda",
        "device",
        "log_level",
        "api_key",
        "base_url",
        "model",
        "temperature",
    )

    def __init__(self) -> None:
        """从环境变量读取全部配置项并冻结到实例上。"""
        # ---------- 路径（支持环境变量覆盖，便于多知识库切换）----------
        self.papers_dir = Path(_env_str("PAPERS_DIR", str(PROJECT_ROOT / "papers")))
        self.persist_dir = Path(_env_str("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db")))
        self.embedding_model = _env_str(
            "EMBEDDING_MODEL", str(PROJECT_ROOT / "local_models/BAAI/bge-large-zh-v1.5")
        )

        # ---------- 检索参数 ----------
        # chunk_size=500：中文论文一个自然段约 300~600 字。
        # 切到 800 会跨段混入无关结论；切到 300 会切断"方法→结果"的因果链。
        # 该值由 eval/run_eval.py 的 Recall@K 曲线确定，修改后需重跑评测。
        self.chunk_size = _env_int("CHUNK_SIZE", 500)
        self.chunk_overlap = _env_int("CHUNK_OVERLAP", 50)

        # top_k=8 由消融实验确定，不是拍脑袋选的：
        #   k=1 → Recall 63.6% | k=4 → 72.7% | k=8 → 100% | k=12 → 100%
        # k=8 是性价比拐点：命中率提升到 100%、整体通过率 73.3%→93.3%，
        # 代价是平均延迟 11.3s→18.2s。文献精读场景准确性优先，故选 8。
        # 若追求交互速度，设 RETRIEVE_TOP_K=4（见 README 消融表）。
        self.retrieve_top_k = _env_int("RETRIEVE_TOP_K", 8)
        # fetch_k 是 MMR 重排前的候选池，必须显著大于 top_k，
        # 否则 MMR 会退化成普通相似度检索（取 4 倍以留出多样性选择空间）。
        self.fetch_k = _env_int("FETCH_K", max(20, self.retrieve_top_k * 4))
        self.mmr_lambda = _env_float("MMR_LAMBDA", 0.5)

        # ---------- 运行环境 ----------
        self.device = _env_str("DEVICE", "auto")
        self.log_level = _env_str("LOG_LEVEL", "INFO").upper()

        # ---------- 大模型 API ----------
        self.api_key = _env_str("DEEPSEEK_API_KEY", "")
        self.base_url = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = _env_str("DEEPSEEK_MODEL", "deepseek-chat")
        # temperature=0.1：文献问答要求答案稳定可复现。高温度会让同一次评测
        # 出现随机波动，无法对比优化前后的效果。
        self.temperature = 0.1

    def resolve_device(self) -> str:
        """把 ``auto`` 解析为实际可用设备。

        Returns:
            ``"cuda"`` 或 ``"cpu"``。若 torch 未安装或 CUDA 不可用则回退 ``"cpu"``，
            这样没有显卡的环境无需修改任何代码。
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def validate(self, require_api_key: bool = True) -> None:
        """校验关键配置，把问题暴露在启动阶段。

        Args:
            require_api_key: 是否要求配置 API Key。
                离线构建索引时不需要 Key，可传 ``False`` 跳过该项检查。

        Raises:
            ValueError: 缺少 API Key 或分片参数不合理时抛出，
                错误信息会附带修复方法。
            FileNotFoundError: 文献目录或嵌入模型不存在。
        """
        if require_api_key and not self.api_key:
            hint = (
                f"未找到 .env 文件：{ENV_FILE}"
                if not _ENV_LOADED
                else f".env 已加载（{ENV_FILE}），但其中没有 DEEPSEEK_API_KEY"
            )
            raise ValueError(
                f"未配置 DEEPSEEK_API_KEY。\n"
                f"  原因：{hint}\n"
                f"  修复：复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY=sk-xxxx"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 "
                f"chunk_size({self.chunk_size})，否则分片无法向前推进。"
            )

        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda 应在 [0, 1] 区间，实际为 {self.mmr_lambda}")

        if not self.papers_dir.is_dir():
            raise FileNotFoundError(
                f"文献目录不存在：{self.papers_dir}\n"
                f"  修复：创建该目录并放入 PDF/DOCX 文件，"
                f"或用环境变量 PAPERS_DIR 指定其他路径。"
            )

        if not Path(self.embedding_model).exists():
            raise FileNotFoundError(
                f"嵌入模型不存在：{self.embedding_model}\n"
                f"  修复：从 HuggingFace 下载 BAAI/bge-large-zh-v1.5 到该目录，"
                f"或设置 EMBEDDING_MODEL 为模型名称让其自动下载。"
            )

    @property
    def env_loaded(self) -> bool:
        """``.env`` 文件是否成功加载（便于诊断配置问题）。"""
        return _ENV_LOADED

    def summary(self) -> dict[str, object]:
        """返回可安全打印的配置摘要（不含密钥明文）。"""
        return {
            "文献目录": str(self.papers_dir),
            "向量库": str(self.persist_dir),
            "嵌入模型": Path(self.embedding_model).name,
            "运行设备": self.resolve_device(),
            "分片大小": f"{self.chunk_size} / 重叠 {self.chunk_overlap}",
            "检索数量": f"top_k={self.retrieve_top_k}, fetch_k={self.fetch_k}, "
            f"mmr_lambda={self.mmr_lambda}",
            "大模型": f"{self.model} @ {self.base_url}",
            "API Key": f"已配置（{len(self.api_key)} 字符）" if self.api_key else "!! 未配置",
            ".env 加载": "成功" if _ENV_LOADED else f"!! 失败（未找到 {ENV_FILE}）",
        }

    def __repr__(self) -> str:
        """返回简短可读的配置摘要（调试时不必展开全部字段）。"""
        return (
            f"Settings(model={self.model!r}, device={self.resolve_device()!r}, "
            f"chunk_size={self.chunk_size}, top_k={self.retrieve_top_k})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一配置实例（带缓存）。

    Returns:
        Settings: 配置对象，进程内只构造一次。
    """
    return Settings()


if __name__ == "__main__":
    # 诊断入口：python config.py 直接查看当前生效配置
    settings = get_settings()
    print("当前配置：")
    for key, value in settings.summary().items():
        print(f"  {key:12} {value}")

    print("\n配置校验：", end="")
    try:
        settings.validate(require_api_key=False)
        print("通过（文件与目录齐备）")
    except (ValueError, FileNotFoundError) as exc:
        print(f"失败\n  {exc}")
