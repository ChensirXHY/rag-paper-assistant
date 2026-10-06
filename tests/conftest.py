"""pytest 全局配置与共享 fixture。

设计原则（与 CI 强相关，改动前请先读完）：

1. **单元测试零外部依赖**：不读真实 ``papers/`` 目录、不加载嵌入模型、
   不要求 API Key、不联网。全部用临时目录造数据、用 ``monkeypatch`` 打桩。
   这样 CI 里只装几个轻量包就能在秒级跑完，不需要 torch / chromadb。
2. **真实资源测试单独标记**：需要真实 PDF、嵌入模型或向量库的测试一律加
   ``@pytest.mark.integration``，默认被本文件跳过，本地用
   ``pytest tests/ -v --run-integration`` 显式开启。
3. **宁可失败也不要"假装通过"**：这里只有"缺第三方包"导致的 skip
   （``pytest.importorskip``），没有任何为了绕开断言而写的跳过逻辑。

关于 ``sys.path``：本项目是"脚本式布局"（``config.py`` 在根目录、
业务代码在 ``src/``），不是可安装包。因此这里显式把项目根目录插到
``sys.path`` 首位，保证从任何工作目录运行 ``pytest`` 都能
``import config`` / ``import src.xxx``。

关于临时目录（重要，别改回 ``tmp_path``）：
    本项目用自己的 :func:`tmp_dir` fixture 代替 pytest 内置的 ``tmp_path``。
    原因不是洁癖，而是踩过坑：``tmp_path`` 在每次创建目录前会清理
    **上一次运行**留下的目录，只要那一步失败（受限沙箱 / 只读 CI 容器 /
    别人留下的只读垃圾目录），所有用到 ``tmp_path`` 的测试都会以
    ``PermissionError`` 直接 ERROR —— 与测试代码对不对毫无关系，
    而且报错信息指向的是临时目录，非常难定位。

    :func:`tmp_dir` 用 ``tempfile.mkdtemp`` 在项目内建目录，只依赖
    "能创建目录"这一个前提，标准环境与受限环境行为一致。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# 环境准备：必须在任何业务模块被 import 之前完成
# ----------------------------------------------------------------------
# tests/conftest.py → tests/ → 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 假 Key：让 `Settings.validate()` 这类"配置自检"逻辑能在无网络环境通过。
# 只在未设置时兜底，绝不覆盖开发者本机 .env 里的真实配置。
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")

# 真实项目副本（含 papers/ 与 local_models/），仅集成测试使用。
# 不存在时集成测试会被跳过，而不是报错 —— 换机器/CI 上不必准备 1.3 GB 模型。
STAGING_ROOT = Path(r"E:\project\_staging_test")

# 临时目录统一放这里：项目内、可预测、便于失败后翻查现场。
# 为什么不叫 .pytest_tmp：pytest 内置 tmp_path 会创建同名目录并给它打上
# 更严格的权限（0o700），在受限环境里那个目录之后连自己都写不进去。
# 用一个专属名字可以彻底避开这层耦合。
TMP_ROOT_NAME = ".tmp_tests"


# ----------------------------------------------------------------------
# 自定义命令行参数与 marker
# ----------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """注册 ``--run-integration`` 开关。

    用 ``hasattr`` 保护：pytest 加载 conftest 的路径不止一条（根目录、
    插件入口、xdist 子进程），重复注册同名选项会直接报错。
    """
    if hasattr(parser, "_dsh_integration_option_added"):
        return
    group = parser.getgroup("rag", "文献 RAG 项目测试选项")
    group.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行需要真实 PDF / 嵌入模型 / 向量库的集成测试（默认跳过）",
    )
    parser._dsh_integration_option_added = True  # type: ignore[attr-defined]


def pytest_configure(config: pytest.Config) -> None:
    """注册 ``integration`` marker。

    不做这一步时 pytest 会对 ``@pytest.mark.integration`` 发
    ``PytestUnknownMarkWarning``，在 ``-W error`` 下会直接变成失败。
    """
    config.addinivalue_line(
        "markers",
        "integration: 需要真实 PDF、嵌入模型或向量库的集成测试，"
        "默认跳过，用 --run-integration 开启",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """未传 ``--run-integration`` 时，跳过所有集成测试。

    实现细节：这里用 ``pytest.mark.skip``（而非 ``pytest.skip``）是为了让
    跳过原因完整地显示在 ``-v`` 输出的 SKIPPED 行里，便于确认
    "到底跳了多少、为什么跳"。
    """
    if config.getoption("--run-integration", default=False):
        return
    skip_integration = pytest.mark.skip(reason="需要真实资源，加 --run-integration 运行")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# ----------------------------------------------------------------------
# 临时目录
# ----------------------------------------------------------------------
_SESSION_TMP_DIRS: list[Path] = []
_SESSION_COUNTERS: dict[str, int] = {}


def _usable_tmp_root() -> Path:
    """挑一个能真正读写的临时根目录。

    候选顺序：项目内 ``.tmp_tests`` → 系统临时目录 → 新建独立临时目录。
    每个候选都会**实际建子目录并写一个中文名文件**再清理掉 ——
    受限沙箱里存在"能建目录但不能写文件"或"不能建子目录"的目录，
    不真写一遍就发现不了（这类失败会以 PermissionError 的形式
    淹没几十个测试，且报错完全指不到原因）。

    Returns:
        Path: 可用的临时根目录。
    """
    candidates = [PROJECT_ROOT / TMP_ROOT_NAME]
    if os.environ.get("TEMP"):
        candidates.append(Path(os.environ["TEMP"]))
    candidates.append(Path(tempfile.gettempdir()))

    for base in candidates:
        probe_dir = base / "probe"
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
            (probe_dir / "中文文件名.txt").write_text("ok", encoding="utf-8")
            shutil.rmtree(probe_dir)
        except OSError:
            continue
        return base

    # 兜底：交给 tempfile 在系统临时区里新建一个独立目录
    return Path(tempfile.mkdtemp(prefix="rag-tests-"))


@pytest.fixture
def tmp_dir(request: pytest.FixtureRequest) -> Path:
    """给单个测试提供一个干净的临时目录。

    与 pytest 内置 ``tmp_path`` 的区别（以及为什么不用 ``tmp_path``）：

        * **目录名可预测**：用"测试名 + 序号"命名，而不是随机串。
          除了便于失败后翻查现场，还避开了一个真实存在的坑 ——
          受限沙箱会拒绝写入"运行时生成的随机名目录"
          （表现为 ``PermissionError [Errno 13]``，且报错完全指不到原因）。
        * **不依赖 pytest 的 basetemp 机制**：那套机制在创建目录前会先清理
          上一次运行留下的目录，清理失败会让所有用 ``tmp_path`` 的测试
          直接 ERROR。

    Args:
        request: pytest 的 fixture 请求对象，用来取当前测试名。

    Returns:
        Path: 该测试独占的空目录（已创建）。
    """
    root = _usable_tmp_root()
    safe_name = re.sub(r"[^0-9A-Za-z_\-]", "_", request.node.name)[:60] or "test"
    _SESSION_COUNTERS[safe_name] = _SESSION_COUNTERS.get(safe_name, 0) + 1
    path = root / f"{safe_name}-{_SESSION_COUNTERS[safe_name]}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    _SESSION_TMP_DIRS.append(path)
    try:
        yield path
    finally:
        # 清理失败不影响测试结论（可能是文件仍被占用），但会留下现场便于排查
        shutil.rmtree(path, ignore_errors=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """整个会话结束后清掉本次创建的临时目录。

    测试数据都是当场生成的，保留它们没有意义；全部清理可以避免
    ``.tmp_tests`` 随运行次数无限膨胀。
    """
    for path in list(_SESSION_TMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)
    _SESSION_TMP_DIRS.clear()
    tmp_root = PROJECT_ROOT / TMP_ROOT_NAME
    try:
        if tmp_root.is_dir() and not any(tmp_root.iterdir()):
            tmp_root.rmdir()
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_global_state():
    """每个测试前后清理模块级缓存，避免测试之间互相污染。

    清理对象：
        * ``config.get_settings`` 的 ``lru_cache``：它缓存了第一个构造出来的
          Settings，会让后面改了环境变量的测试拿到旧配置；
        * ``src.logger`` 的 ``_configured`` 标记：让每个测试都能重新按需
          初始化日志，而不是因为"已经配置过"被静默跳过。
    """
    import config
    from src import logger as logger_module

    config.get_settings.cache_clear()
    was_configured = logger_module._configured
    yield
    config.get_settings.cache_clear()
    logger_module._configured = was_configured


# ----------------------------------------------------------------------
# 通用 fixture
# ----------------------------------------------------------------------
@pytest.fixture
def project_root() -> Path:
    """返回项目根目录（供需要显式路径的测试使用）。"""
    return PROJECT_ROOT


@pytest.fixture
def settings(tmp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """返回一个真实的 :class:`config.Settings` 实例。

    为什么用环境变量而不是直接给对象赋值：``Settings`` 用 ``__slots__``
    且没有 setter，字段在 ``__init__`` 里从环境变量读取一次就冻结 ——
    这正是"配置不可变"的设计意图。所以测试必须像真实启动那样，
    先准备好环境变量再构造实例。

    默认把 ``papers_dir`` / ``persist_dir`` / ``embedding_model`` 全部指向
    临时目录，并**预先创建好**：``validate()`` 要求"文献目录存在、
    嵌入模型路径存在"，否则任何调用 validate 的代码（如
    ``VectorStoreManager.sync``）都会在测试里抛 FileNotFoundError。

    Args:
        tmp_dir: 本次测试独占的临时目录。
        monkeypatch: 用于设置并自动还原环境变量。

    Returns:
        Settings: 指向临时目录的真实配置对象。
    """
    from config import Settings

    papers = tmp_dir / "papers"
    papers.mkdir()
    chroma = tmp_dir / "chroma_db"
    model = tmp_dir / "local_model"
    model.mkdir()

    monkeypatch.setenv("PAPERS_DIR", str(papers))
    monkeypatch.setenv("CHROMA_DB_DIR", str(chroma))
    monkeypatch.setenv("EMBEDDING_MODEL", str(model))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")
    return Settings()


@pytest.fixture
def tmp_papers(tmp_dir: Path) -> Path:
    """造一个装着"假文献"的目录。

    只写 ``.txt`` 而不生成 PDF：生成 PDF 需要额外依赖且体积大，
    而本项目所有解析逻辑（清洗、过滤、分片）都在"文本已经读出来"之后，
    用 txt 就能完整覆盖，没必要为了逼真牺牲 CI 速度。

    目录结构（同时覆盖"正常文件"与"应被跳过的垃圾文件"）::

        tmp_papers/
        ├── 论文一.txt          # 正常中文段落
        ├── 论文二.md           # 短文件
        ├── ~$论文一.txt        # Office 临时文件，必须被跳过
        ├── .hidden.txt         # 隐藏文件，必须被跳过
        └── 说明.pdf            # 扩展名受支持但内容不是 PDF（解析失败路径）

    Returns:
        Path: 假文献目录。
    """
    papers = tmp_dir / "papers"
    papers.mkdir()

    (papers / "论文一.txt").write_text(
        "本文提出了一种基于注意力机制的光伏功率预测模型。"
        "该方法在数据预处理阶段使用了变分模态分解来抑制噪声。"
        "实验结果表明，所提模型在晴天工况下的均方根误差下降了百分之十二。\n"
        "为了验证泛化能力，作者还在阴天与雨天工况下做了对比实验。\n",
        encoding="utf-8",
    )
    (papers / "论文二.md").write_text(
        "# 光伏功率预测综述\n\n短期预测的主流做法是数值天气预报加机器学习。\n",
        encoding="utf-8",
    )

    # 下面两个文件必须被 iter_source_files 过滤掉，否则会污染索引
    (papers / "~$论文一.txt").write_text("Office 编辑临时文件，内容是垃圾", encoding="utf-8")
    (papers / ".hidden.txt").write_text("隐藏文件，不应被索引", encoding="utf-8")
    # 受支持的扩展名 + 非法内容：用于验证"单个文件解析失败不中断批量导入"
    (papers / "说明.pdf").write_bytes(b"this is definitely not a pdf file")
    return papers


@pytest.fixture
def fake_documents():
    """构造两个伪造的 Document，模拟 loader 的输出。

    metadata 形状与 :func:`src.loader.load_file` 的真实输出保持一致
    （``source`` 为绝对路径字符串、``page`` 为页码或 0、``file_name``、
    ``file_type``、``fingerprint``），这样下游断言才有意义。
    """
    from langchain_core.documents import Document

    return [
        Document(
            page_content="本文提出 TCN-BiLSTM-Attention-ESN 混合模型用于光伏功率预测。"
            "该模型先用 TCN 提取局部时序特征，再用 BiLSTM 捕捉长程依赖。",
            metadata={
                "source": r"E:\papers\论文一.pdf",
                "page": 0,
                "file_name": "论文一.pdf",
                "file_type": "pdf",
                "fingerprint": "论文一.pdf@1024:1700000000",
            },
        ),
        Document(
            page_content="为了验证模型有效性，本文在 2022 年全年数据集上做了消融实验，"
            "并与持久化模型、BP 神经网络做了对比。",
            metadata={
                "source": r"E:\papers\论文二.pdf",
                "page": 1,
                "file_name": "论文二.pdf",
                "file_type": "pdf",
                "fingerprint": "论文二.pdf@2048:1700000001",
            },
        ),
    ]


@pytest.fixture
def staging_project() -> Path:
    """返回真实项目副本路径；不存在时跳过测试。

    只给集成测试用。这里显式判存在再 skip，是为了让"没准备真实数据"的
    环境给出清晰的跳过原因，而不是让人对着一堆 FileNotFoundError 排查。
    """
    if not STAGING_ROOT.is_dir():
        pytest.skip(f"未找到真实项目副本：{STAGING_ROOT}")
    return STAGING_ROOT
