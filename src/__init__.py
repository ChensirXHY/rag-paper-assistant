"""文献智能问答系统 —— 核心模块包。

模块划分：
    - :mod:`src.loader`       多格式文档加载（PDF / DOCX / TXT / MD）
    - :mod:`src.splitter`     中文优化的文本分片
    - :mod:`src.vectorstore`  Chroma 向量库与增量索引
    - :mod:`src.qa`           RAG 检索问答编排
    - :mod:`src.logger`       统一日志
"""

from __future__ import annotations

__version__ = "0.2.0"
