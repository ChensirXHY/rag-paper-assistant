"""统一日志模块。

为什么不用 ``print``：
    1. print 无法分级 —— 调试信息会混进正式输出；
    2. print 无法关闭 —— 模块被 import 时副作用不可控；
    3. print 无法落盘 —— 出问题后没有现场可查。

本模块让全项目输出形如::

    2026-10-06 20:35:12 | INFO     | src.loader | 文献加载完成：6 个文件、67 页
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False

# langchain-community 已被官方标记为 sunset（不再积极维护），
# 但本项目仍在使用它的 Chroma 与 HuggingFaceBgeEmbeddings 集成。
# 这里过滤掉它的 DeprecationWarning，避免刷屏淹没真正的警告。
# 后续迁移方向见 README「已知限制与后续计划」。
_NOISY_PATTERNS = (
    r".*langchain-community.*sunset.*",
    r".*langchain_community.*deprecated.*",
)


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> None:
    """初始化根 logger。进程内只需调用一次，重复调用会被忽略。

    Args:
        level: 日志级别，如 ``"DEBUG"``/``"INFO"``/``"WARNING"``。
        log_to_file: 是否同时写入 ``logs/app.log``。
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())

    # 清掉可能已存在的 handler，避免重复输出（例如在 Jupyter 中反复执行）
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        # encoding="utf-8" 是必须的：否则中文日志在 Windows 上会抛 UnicodeEncodeError
        file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8", mode="a")
        file_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    # 第三方库日志过于啰嗦，压到 WARNING 以保留有效信息
    for noisy in (
        "httpx",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "matplotlib",
        "transformers",
        "huggingface_hub",
        "filelock",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for pattern in _NOISY_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取带统一格式的 logger。

    Args:
        name: 通常直接传 ``__name__``，这样日志能定位到具体模块。

    Returns:
        配置好的 Logger 实例。

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("加载完成，共 %d 页", 67)
    """
    setup_logging()
    return logging.getLogger(name)
