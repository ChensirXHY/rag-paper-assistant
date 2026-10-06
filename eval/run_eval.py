"""检索问答质量评测脚本。

把"回答质量提升了"从主观感受变成可复现的数字，这是本脚本存在的唯一理由。
没有它，任何一次提示词/参数改动都只能靠"感觉好像好点了"来判断，
也无法回答"chunk_size 从 500 改成 300 到底是变好还是变坏"。

指标定义（每一项都刻意选成"可解释、可复现、改动后方向明确"）：

    检索命中率 Recall@K
        定义：``expected_source``（期望文献的文件名）出现在 ``retrieve()``
        返回块的来源里的问题占比。
        为什么这么定义：检索是 RAG 的第一道关口。检索没拿到正确文献，
        后面的生成再强也只能编答案。用"文件名命中"而不是"块 id 命中"，
        是因为块 id 会随 chunk_size 变化 —— 换分片参数后指标就不可比了。
        该指标**不需要 API Key**，因此模型接口不可用时也能跑（见 --retrieval-only）。

    出处准确率
        定义：答案里 ``[片段N]`` 引用编号中，N 落在本次实际检索块数量范围内的比例
        （按引用条目加权，再对问题取平均）。
        为什么这么定义：``[片段9]`` 这种越界引用是典型的"看起来有出处、
        实际是编的"，前端点进去会指向不存在的片段。它衡量的是"引用是否可用"。

    拒答准确率
        定义：``type=absent`` 的问题中，回答被判定为"明确表示文献未提及"的比例。
        为什么这么定义：文献库里没有的内容，唯一正确的回答就是拒答。
        这条指标专门用来量化幻觉。

    幻觉率
        定义：``1 - 拒答准确率``。
        为什么单列：它是"加法指标"（越低越好），放进 README 比"拒答率 0.75"
        更能传达风险；同时这也呼应了项目里"宁可说不知道，也不要编"的产品选择。

    整体通过率
        定义：按问题类型分别判定后取平均（判定规则见 :func:`judge`）。
        为什么不用单一"正确率"：``fact`` 看关键词、``compare`` 看是否给出
        多来源综合、``absent`` 看是否拒答 —— 三类问题的成功标准本来就不同，
        混在一起会让优化方向变得模糊。

    平均延迟 / P95 延迟
        定义：``Answer.latency_s`` 的均值与第 95 百分位（无 numpy 时退化为最大值）。
        为什么要 P95：均值会被少数快请求掩盖，用户真正抱怨的是"偶尔要等半分钟"
        这种长尾体验。

用法::

    # 只跑检索指标（不需要 API Key，秒级出结果，适合 CI 与日常回归）
    python eval/run_eval.py --retrieval-only

    # 全量评测（需要有效的 DEEPSEEK_API_KEY）
    python eval/run_eval.py --limit 5

    # 消融实验：比较不同 top_k 对召回的影响
    python eval/run_eval.py --retrieval-only --top-k 1
    python eval/run_eval.py --retrieval-only --top-k 4 --report reports/topk4.md

    # 导出明细，便于逐题排查"到底哪一条没答对"
    python eval/run_eval.py --json reports/detail.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 允许 "python eval/run_eval.py" 这种直接执行的方式：
# 脚本方式运行时 sys.path[0] 是 eval/ 而不是项目根目录，
# 不补这一行就会 ImportError: No module named 'config'。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings, get_settings  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.qa import PaperRAG, collect_sources  # noqa: E402

logger = get_logger(__name__)

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.jsonl"

# 判定"模型明确拒答"的关键词。
# 选取原则：只收"明确表示文献里没有"的表述，不收"可能/也许"这类模糊词 ——
# 否则一句"文献中可能没有提到"会被当成正确拒答，反而鼓励模型说废话。
# 提示词 prompts/qa_prompt.txt 里也明确要求使用"未提及/未涉及"这类措辞，
# 两边是配套的：改提示词时记得同步这里。
REFUSAL_KEYWORDS: tuple[str, ...] = (
    "未提及",
    "未涉及",
    "没有提及",
    "没有提到",
    "未提到",
    "没有涉及",
    "未包含",
    "不包含",
    "无法回答",
    "无法从",
    "没有相关",
    "未找到相关",
    "文献中没有",
    "文中未",
    "暂无相关",
)

# 回答里出现拒答词后的最大可信长度。
# 为什么需要这个上限：模型有时会先说一句"文献未提及"，再补一段自己编的
# "不过一般来说光伏电站通常……"。这种回答看着礼貌，实际就是幻觉，
# 不能算作成功拒答。
REFUSAL_MAX_CHARS = 150

# "先拒答、再补充"的语言标记。
# 实测模型最爱用的句式是"文献未提及。不过一般来说……"，其中"不过/但是/
# 一般来说/通常/建议"这些词就是它开始编的信号。
# 只要出现这类转折或泛化词，即使回答很短也不算成功拒答 ——
# 这条规则与 REFUSAL_MAX_CHARS 配合：长度上限管"长篇补充"，
# 转折词管"短篇补充"（模型三五句话就能编出一个文献里不存在的结论）。
REFUSAL_CONTRAST_MARKERS: tuple[str, ...] = (
    "不过",
    "但是",
    "但一般",
    "然而",
    "一般来说",
    "通常",
    "建议",
    "可以参考",
    "据我所知",
    "根据经验",
    "一般情况下",
)

# 匹配答案里的引用标记 ``[片段3]``
_CITATION_RE = re.compile(r"\[片段\s*(\d+)\]")

# 未指定 --report/--json 时打印到标准输出的表格标题
_METRIC_LABELS = {
    "recall": "检索命中率 Recall@K",
    "refusal": "拒答准确率",
    "hallucination": "幻觉率",
    "citation": "出处准确率",
    "pass_rate": "整体通过率",
    "latency_avg": "平均延迟(s)",
    "latency_p95": "P95 延迟(s)",
}


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class EvalQuestion:
    """一条评测问题。

    Attributes:
        id: 唯一标识，用于在报告和明细 JSON 里定位问题。
        question: 中文问题文本。
        type: ``fact`` / ``compare`` / ``absent``。
        expected_source: 期望命中的文献文件名；``absent`` 类为空串。
        expected_keywords: 期望答案包含的关键词（任一命中即算命中）。
    """

    id: str
    question: str
    type: str = "fact"
    expected_source: str = ""
    expected_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], line_no: int) -> EvalQuestion:
        """从 JSON 对象构造，缺字段时给出定位到行的报错。

        Args:
            raw: 解析出的 JSON 对象。
            line_no: 该行在 questions.jsonl 中的行号（从 1 开始）。

        Returns:
            EvalQuestion 实例。

        Raises:
            ValueError: 缺少必填字段或 ``type`` 取值非法。
        """
        for key in ("id", "question"):
            if not str(raw.get(key, "")).strip():
                raise ValueError(f"questions.jsonl 第 {line_no} 行缺少必填字段 {key!r}")

        qtype = str(raw.get("type", "fact")).strip() or "fact"
        if qtype not in {"fact", "compare", "absent"}:
            raise ValueError(
                f"questions.jsonl 第 {line_no} 行的 type={qtype!r} 非法，"
                "只能是 fact / compare / absent"
            )

        keywords = raw.get("expected_keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]

        return cls(
            id=str(raw["id"]).strip(),
            question=str(raw["question"]).strip(),
            type=qtype,
            expected_source=str(raw.get("expected_source") or "").strip(),
            expected_keywords=[str(k).strip() for k in keywords if str(k).strip()],
        )


@dataclass
class EvalResult:
    """一条问题的评测明细。

    Attributes:
        question: 对应的问题。
        retrieval_ok: ``expected_source`` 是否出现在检索结果中。
            ``expected_source`` 为空时记 ``None``（不计入检索指标）。
        retrieved_sources: 检索到块的文件名（已去重、保序）。
        latency_s: 端到端耗时。
        answer: 模型回答；``--retrieval-only`` 时为空串。
        cited_indices: 答案里出现的引用编号。
        citation_ok: 引用是否全部落在有效范围内；未评价时为 ``None``。
        refused: 是否被判为明确拒答；未评价时为 ``None``。
        keyword_hit: 关键词是否命中；无关键词时为 ``None``。
        error: 调用失败时的错误摘要（失败问题不计入质量指标）。
    """

    question: EvalQuestion
    retrieval_ok: bool | None = None
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_chunks: int = 0
    latency_s: float = 0.0
    answer: str = ""
    cited_indices: list[int] = field(default_factory=list)
    citation_ok: bool | None = None
    refused: bool | None = None
    keyword_hit: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        """该问题是否计入了质量指标（调用失败的问题不计入）。"""
        return not self.error

    @property
    def passed(self) -> bool | None:
        """按问题类型判定是否通过；无法判定时返回 ``None``。"""
        if self.error:
            return None
        if self.question.type == "absent":
            return self.refused
        if self.keyword_hit is None:
            return self.citation_ok
        return bool(self.keyword_hit)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（用于 ``--json`` 明细）。"""
        return {
            "id": self.question.id,
            "question": self.question.question,
            "type": self.question.type,
            "expected_source": self.question.expected_source,
            "retrieval_ok": self.retrieval_ok,
            "retrieved_chunks": self.retrieved_chunks,
            "retrieved_sources": self.retrieved_sources,
            "latency_s": round(self.latency_s, 3),
            "answer": self.answer[:500],
            "cited_indices": self.cited_indices,
            "citation_ok": self.citation_ok,
            "refused": self.refused,
            "keyword_hit": self.keyword_hit,
            "passed": self.passed,
            "error": self.error,
        }


# ----------------------------------------------------------------------
# 问题集加载
# ----------------------------------------------------------------------
def load_questions(path: Path) -> list[EvalQuestion]:
    """读取 JSONL 问题集，支持 ``//`` 注释行与空行。

    为什么手写解析而不是直接用 ``json.loads`` 逐行：问题集是给人维护的资产，
    需要写注释说明"这条为什么算无法回答型"。标准 JSON 不支持注释，
    而 JSONL 里加一层过滤的成本远低于维护一个单独的说明文档。

    Args:
        path: questions.jsonl 路径。

    Returns:
        问题列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 某行不是合法 JSON 或缺少必填字段。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"问题集不存在：{path}\n  修复：确认 eval/questions.jsonl 是否在版本库中。"
        )

    questions: list[EvalQuestion] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"questions.jsonl 第 {line_no} 行不是合法 JSON：{exc.msg}") from exc
        questions.append(EvalQuestion.from_dict(data, line_no))

    if not questions:
        raise ValueError(f"{path} 中没有有效问题（是否全被注释掉了？）")

    ids = [q.id for q in questions]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise ValueError(f"questions.jsonl 存在重复 id：{sorted(duplicated)}")

    return questions


# ----------------------------------------------------------------------
# 判定逻辑（纯函数，单独可测）
# ----------------------------------------------------------------------
def is_refusal(answer: str) -> bool:
    """判断回答是否明确表示"文献中未提及"。

    判定规则（三条同时满足）：
        1. 出现 :data:`REFUSAL_KEYWORDS` 中的任一关键词；
        2. 回答长度不超过 :data:`REFUSAL_MAX_CHARS`，且不是空串；
        3. 不出现 :data:`REFUSAL_CONTRAST_MARKERS` 里的转折/泛化词。

    后两条是刻意的：模型常见"先拒答、再补充一般性知识"的写法，
    后半段往往是编的（"文献未提及。不过一般来说，光伏电站通常……"）。
    只看到"未提及"就判通过，会把幻觉记成"正确拒答"，让幻觉率虚低 ——
    那比指标难看糟糕得多。

    Args:
        answer: 模型回答文本。

    Returns:
        ``True`` 表示可视为成功拒答。
    """
    text = (answer or "").strip()
    if not text:
        return False
    if len(text) > REFUSAL_MAX_CHARS:
        return False
    if any(marker in text for marker in REFUSAL_CONTRAST_MARKERS):
        return False
    return any(keyword in text for keyword in REFUSAL_KEYWORDS)


def parse_citations(answer: str) -> list[int]:
    """抽取答案里的 ``[片段N]`` 引用编号（按出现顺序，保留重复）。

    Args:
        answer: 模型回答文本。

    Returns:
        引用编号列表，例如 ``[1, 2, 2]``。
    """
    return [int(m.group(1)) for m in _CITATION_RE.finditer(answer or "")]


def judge_citation(cited: list[int], retrieved_chunks: int) -> bool | None:
    """判断引用是否全部有效（编号落在实际检索到的片段范围内）。

    Args:
        cited: 答案里的引用编号。
        retrieved_chunks: 本次实际检索到的块数（即提示词里编号的最大值）。

    Returns:
        全部有效返回 ``True``；存在越界或一个引用都没有返回 ``False``；
        ``retrieved_chunks`` 为 0（没检索到内容）时返回 ``None``（无法评价）。
    """
    if retrieved_chunks <= 0:
        return None
    if not cited:
        return False
    return all(1 <= idx <= retrieved_chunks for idx in cited)


def judge_keywords(answer: str, keywords: list[str]) -> bool | None:
    """判断答案是否命中任一期望关键词（大小写不敏感）。

    Args:
        answer: 模型回答文本。
        keywords: 期望关键词列表。

    Returns:
        有命中返回 ``True``，未命中返回 ``False``；关键词为空时返回 ``None``。
    """
    if not keywords:
        return None
    lowered = (answer or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def percentile(values: list[float], ratio: float) -> float:
    """用最近秩法计算分位数（不依赖 numpy，CI 里少一个依赖）。

    Args:
        values: 数值列表，空列表返回 0。
        ratio: 分位比例，例如 ``0.95``。

    Returns:
        分位数值。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


# ----------------------------------------------------------------------
# 评测主流程
# ----------------------------------------------------------------------
def evaluate_retrieval(
    rag: PaperRAG, questions: list[EvalQuestion], top_k: int | None
) -> list[EvalResult]:
    """只跑检索，不调用大模型。

    Args:
        rag: 已初始化的问答系统（其向量库应已同步）。
        questions: 问题列表。
        top_k: 覆盖检索数量；``None`` 时用配置值。

    Returns:
        评测明细列表（``latency_s`` 为纯检索耗时）。
    """
    results: list[EvalResult] = []
    for index, question in enumerate(questions, 1):
        started = time.perf_counter()
        try:
            docs = rag.retrieve(question.question, top_k=top_k)
        except Exception as exc:  # 单条失败不应中断整轮评测
            logger.warning("检索失败 [%s]：%s", question.id, exc)
            results.append(EvalResult(question=question, error=f"检索失败：{exc}"))
            continue

        elapsed = time.perf_counter() - started
        sources: list[str] = []
        for source in collect_sources(docs):
            if source.file not in sources:
                sources.append(source.file)

        retrieval_ok: bool | None = None
        if question.expected_source:
            retrieval_ok = question.expected_source in sources

        results.append(
            EvalResult(
                question=question,
                retrieval_ok=retrieval_ok,
                retrieved_sources=sources,
                retrieved_chunks=len(docs),
                latency_s=elapsed,
            )
        )
        status = "-" if retrieval_ok is None else ("命中" if retrieval_ok else "未命中")
        logger.info(
            "[%d/%d] %s 检索 %s（%d 块，%.2fs）",
            index,
            len(questions),
            question.id,
            status,
            len(docs),
            elapsed,
        )
    return results


def evaluate_answers(rag: PaperRAG, results: list[EvalResult], top_k: int | None) -> None:
    """在检索结果基础上补跑问答，就地填充质量字段。

    Args:
        rag: 问答系统。
        results: :func:`evaluate_retrieval` 产出的明细（会被就地修改）。
        top_k: 覆盖检索数量。
    """
    for index, result in enumerate(results, 1):
        if result.error:
            continue
        question = result.question
        try:
            answer = rag.ask(question.question, top_k=top_k)
        except Exception as exc:  # 401/超时等：记录后继续跑完剩下的问题
            logger.warning("问答失败 [%s]：%s", question.id, exc)
            result.error = _brief_error(exc)
            continue

        result.answer = answer.text
        result.latency_s = answer.latency_s
        result.cited_indices = parse_citations(answer.text)
        # 用**本次回答实际引用的来源数**做上界：ask() 内部的检索与上面
        # evaluate_retrieval 是两次独立调用，块数可能因库更新而不同。
        upper_bound = max(len(answer.sources), answer.retrieved)
        result.citation_ok = judge_citation(result.cited_indices, upper_bound)
        result.refused = is_refusal(answer.text) if question.type == "absent" else None
        result.keyword_hit = judge_keywords(answer.text, question.expected_keywords)
        logger.info(
            "[%d/%d] %s 问答完成（引用 %d 处，%s，%.2fs）",
            index,
            len(results),
            question.id,
            len(result.cited_indices),
            "通过" if result.passed else "未通过",
            answer.latency_s,
        )


def _brief_error(exc: Exception) -> str:
    """把异常压成一行摘要（完整堆栈留在日志里，明细 JSON 只放结论）。"""
    text = str(exc).replace("\n", " ").strip()
    return text[:200] if text else type(exc).__name__


def summarize(results: list[EvalResult]) -> dict[str, float | int | None]:
    """汇总指标。

    Args:
        results: 评测明细。

    Returns:
        指标字典。无法计算的指标为 ``None``（而不是 0）——
        "没有数据"和"数据是 0 分"必须在报告里区分开。
    """
    scored = [r for r in results if r.ok]
    latencies = [r.latency_s for r in scored]

    with_source = [r for r in scored if r.retrieval_ok is not None]
    recall = (
        sum(1 for r in with_source if r.retrieval_ok) / len(with_source) if with_source else None
    )

    absent = [r for r in scored if r.question.type == "absent" and r.refused is not None]
    refusal = sum(1 for r in absent if r.refused) / len(absent) if absent else None

    cited = [r for r in scored if r.citation_ok is not None]
    citation = sum(1 for r in cited if r.citation_ok) / len(cited) if cited else None

    judged = [r.passed for r in scored if r.passed is not None]
    pass_rate = sum(1 for p in judged if p) / len(judged) if judged else None

    return {
        "total": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "recall": recall,
        "refusal": refusal,
        "hallucination": None if refusal is None else 1.0 - refusal,
        "citation": citation,
        "pass_rate": pass_rate,
        "latency_avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95": percentile(latencies, 0.95),
    }


def group_by_type(results: list[EvalResult]) -> dict[str, dict[str, float | int | None]]:
    """按问题类型分组统计，用来定位"是哪一类问题拖低了指标"。

    ``通过率`` 在"该类问题一个都没评价"时为 ``None``（显示成 N/A）：
    ``--retrieval-only`` 模式下没有答案，所有类型的通过率都无从计算，
    此时显示 0.0% 会让人误以为"全军覆没"。

    Args:
        results: 评测明细。

    Returns:
        ``{类型: {"总数": n, "通过": m, "未评价": k, "通过率": r 或 None}}``。
    """
    groups: dict[str, dict[str, float | int | None]] = {}
    for result in results:
        bucket = groups.setdefault(
            result.question.type, {"总数": 0, "通过": 0, "未评价": 0, "通过率": None}
        )
        bucket["总数"] = int(bucket["总数"]) + 1
        verdict = result.passed
        if verdict is None:
            bucket["未评价"] = int(bucket["未评价"]) + 1
        elif verdict:
            bucket["通过"] = int(bucket["通过"]) + 1
    for bucket in groups.values():
        scored = int(bucket["总数"]) - int(bucket["未评价"])
        bucket["通过率"] = int(bucket["通过"]) / scored if scored else None
    return groups


# ----------------------------------------------------------------------
# 报告输出
# ----------------------------------------------------------------------
def _fmt(value: float | int | None, percent: bool = False, digits: int = 2) -> str:
    """格式化指标值：``None`` 显示为 ``N/A``（与 0 区分开）。"""
    if value is None:
        return "N/A"
    if percent:
        return f"{float(value) * 100:.1f}%"
    return f"{float(value):.{digits}f}"


def render_markdown(
    summary: dict[str, float | int | None],
    results: list[EvalResult],
    settings: Settings,
    top_k: int,
    retrieval_only: bool,
) -> str:
    """生成可直接粘贴进 README 的 Markdown 报告。

    Args:
        summary: :func:`summarize` 的输出。
        results: 评测明细。
        settings: 当前配置（写进报告便于复现）。
        top_k: 本次使用的检索数量。
        retrieval_only: 是否为"只跑检索"模式。

    Returns:
        Markdown 文本。
    """
    mode = "仅检索指标（--retrieval-only）" if retrieval_only else "检索 + 问答"
    lines = [
        "## 评测结果",
        "",
        f"- 运行模式：**{mode}**",
        f"- 问题数：{summary['total']}（有效 {summary['scored']}"
        + (f"，失败 {summary['failed']}" if summary["failed"] else "")
        + "）",
        f"- 检索参数：top_k={top_k}，chunk_size={settings.chunk_size}，"
        f"chunk_overlap={settings.chunk_overlap}",
        f"- 向量库：`{settings.persist_dir}`",
        f"- 生成模型：`{settings.model}`（问答指标依赖它，检索指标不依赖）",
        "",
        "| 指标 | 数值 | 说明 |",
        "| --- | --- | --- |",
        f"| 检索命中率 Recall@{top_k} | {_fmt(summary['recall'], percent=True)} "
        "| 期望文献出现在检索结果中的比例 |",
        f"| 出处准确率 | {_fmt(summary['citation'], percent=True)} "
        "| 答案中 `[片段N]` 引用全部落在有效范围内的比例 |",
        f"| 拒答准确率 | {_fmt(summary['refusal'], percent=True)} "
        "| 无法回答型问题中，明确回答「未提及」的比例 |",
        f"| 幻觉率 | {_fmt(summary['hallucination'], percent=True)} "
        "| `1 - 拒答准确率`，越低越好 |",
        f"| 整体通过率 | {_fmt(summary['pass_rate'], percent=True)} "
        "| 按问题类型判定后的平均通过率 |",
        f"| 平均延迟 | {_fmt(summary['latency_avg'])} s | 单次请求端到端耗时均值 |",
        f"| P95 延迟 | {_fmt(summary['latency_p95'])} s | 95% 请求不超过该耗时 |",
        "",
    ]

    groups = group_by_type(results)
    if groups:
        lines += [
            "### 分类型通过率",
            "",
            "| 类型 | 总数 | 通过 | 通过率 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for qtype in sorted(groups):
            bucket = groups[qtype]
            lines.append(
                f"| {qtype} | {bucket['总数']} | {bucket['通过']} "
                f"| {_fmt(bucket['通过率'], percent=True)} |"
            )
        lines.append("")

    if retrieval_only:
        missed = [r for r in results if r.retrieval_ok is False]
        lines += [
            "> 本模式不调用大模型，因此出处准确率/拒答准确率/幻觉率显示为 N/A。",
            "> 配置好有效的 `DEEPSEEK_API_KEY` 后去掉 `--retrieval-only` 即可得到完整指标。",
            "",
        ]
        if missed:
            lines += [
                f"### 未命中的问题（{len(missed)} 条，按 top_k 调参的主要抓手）",
                "",
                "| id | 问题 | 期望文献 | 实际召回 |",
                "| --- | --- | --- | --- |",
            ]
            for result in missed:
                actual = "、".join(result.retrieved_sources) or "（无结果）"
                lines.append(
                    f"| {result.question.id} | {result.question.question} "
                    f"| {result.question.expected_source} | {actual} |"
                )
            lines.append("")

    failed = [r for r in results if r.error]
    if failed:
        lines += [
            f"### 调用失败的问题（{len(failed)} 条，未计入指标）",
            "",
            "| id | 错误 |",
            "| --- | --- |",
        ]
        for result in failed:
            lines.append(f"| {result.question.id} | {result.error} |")
        lines.append("")

    return "\n".join(lines)


def render_console(summary: dict[str, float | int | None], top_k: int, retrieval_only: bool) -> str:
    """生成控制台/最终答复里可直接引用的紧凑表格。"""
    lines = [
        f"评测完成（模式：{'仅检索' if retrieval_only else '检索+问答'}，top_k={top_k}）",
        f"问题数 {summary['total']}｜有效 {summary['scored']}｜失败 {summary['failed']}",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
    ]
    for key in ("recall", "citation", "refusal", "hallucination", "pass_rate"):
        is_percent = key in {"recall", "citation", "refusal", "hallucination", "pass_rate"}
        lines.append(f"| {_METRIC_LABELS[key]} | {_fmt(summary[key], percent=is_percent)} |")
    lines.append(f"| {_METRIC_LABELS['latency_avg']} | {_fmt(summary['latency_avg'])} |")
    lines.append(f"| {_METRIC_LABELS['latency_p95']} | {_fmt(summary['latency_p95'])} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 参数列表；``None`` 时取 ``sys.argv[1:]``（便于测试时传入）。

    Returns:
        解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="文献 RAG 问答质量评测（检索指标不需要 API Key）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python eval/run_eval.py --retrieval-only          # 只算检索指标（无需 Key）\n"
            "  python eval/run_eval.py --retrieval-only --top-k 8   # 消融：加大召回\n"
            "  python eval/run_eval.py --limit 3                 # 只跑前 3 条（省时间/省钱）\n"
            "  python eval/run_eval.py --report reports/eval.md  # 输出可粘贴的 Markdown\n"
            "  python eval/run_eval.py --json reports/detail.json   # 导出逐题明细\n"
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"问题集路径（默认 {DEFAULT_QUESTIONS}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="覆盖检索数量（消融实验用；默认取 .env 的 RETRIEVE_TOP_K）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条问题")
    parser.add_argument("--report", type=Path, default=None, help="把 Markdown 报告写入该文件")
    parser.add_argument("--json", type=Path, default=None, help="把逐题明细写入该 JSON 文件")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只跑检索指标：不调用大模型，因此不需要 API Key，秒级出结果",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="评测前全量重建索引（改过分片参数时需要）"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """评测主流程。

    Args:
        argv: 命令行参数；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        进程退出码：0 正常结束，1 环境/配置问题，2 全部问题都失败。
    """
    args = parse_args(argv)

    try:
        questions = load_questions(args.questions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1

    if args.limit is not None and args.limit > 0:
        questions = questions[: args.limit]
    logger.info("载入 %d 条评测问题：%s", len(questions), args.questions)

    settings = get_settings()
    top_k = args.top_k or settings.retrieve_top_k

    # ---- 索引同步（不生成答案，无需 API Key）----
    rag = PaperRAG(settings)
    try:
        stats = rag.ensure_index(rebuild=args.rebuild)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[错误] 索引同步失败：{exc}")
        return 1
    logger.info(
        "索引就绪：%d 个文献、%d 个文本块（本次新增 %d 块，清理 %d 个文件）",
        stats.indexed_files,
        stats.total_chunks,
        stats.added_chunks,
        len(stats.removed_files),
    )

    # ---- 判断能否跑问答 ----
    retrieval_only = args.retrieval_only
    if not retrieval_only and not settings.api_key:
        print(
            "[提示] 未配置 DEEPSEEK_API_KEY，自动降级为 --retrieval-only。\n"
            "       检索指标不受影响；如需问答指标，请在 .env 中填入有效的 Key。"
        )
        retrieval_only = True

    print(
        f"开始评测：{len(questions)} 条问题，top_k={top_k}，"
        f"{'仅检索' if retrieval_only else '检索 + 问答'}\n"
    )

    results = evaluate_retrieval(rag, questions, args.top_k)
    if not retrieval_only:
        evaluate_answers(rag, results, args.top_k)

    summary = summarize(results)
    console_table = render_console(summary, top_k, retrieval_only)
    print("\n" + console_table + "\n")

    markdown = render_markdown(summary, results, settings, top_k, retrieval_only)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown, encoding="utf-8")
        print(f"Markdown 报告已写入：{args.report}")
    else:
        print(markdown)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "top_k": top_k,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "model": settings.model,
                "papers_dir": str(settings.papers_dir),
                "persist_dir": str(settings.persist_dir),
                "retrieval_only": retrieval_only,
            },
            "summary": summary,
            "by_type": group_by_type(results),
            "results": [r.to_dict() for r in results],
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细 JSON 已写入：{args.json}")

    if summary["scored"] == 0:
        print("[错误] 所有问题都失败了，请检查上方日志（常见原因：API Key 失效、嵌入模型缺失）。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
