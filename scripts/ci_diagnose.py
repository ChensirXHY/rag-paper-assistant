"""CI 环境诊断脚本。

用途：GitHub Actions 与本地环境不一致时，先跑这个脚本把"环境长什么样"
和"哪些测试会被跳过/报错"一次性打印出来，避免靠猜。

本地也能用：
    python scripts/ci_diagnose.py
"""

from __future__ import annotations

import importlib
import locale
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("=" * 72)
print("1. 运行环境")
print("=" * 72)
print(f"  Python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"  平台        : {platform.platform()}")
print(f"  文件系统编码: {sys.getfilesystemencoding()}")
print(f"  首选编码    : {locale.getpreferredencoding(False)}")
print(f"  stdout 编码 : {getattr(sys.stdout, 'encoding', '?')}")
print(f"  工作目录    : {os.getcwd()}")
print(f"  项目根目录  : {PROJECT_ROOT}")
print(f"  PYTHONPATH  : {os.environ.get('PYTHONPATH', '(未设置)')}")
print(f"  DEEPSEEK_API_KEY 已设置: {bool(os.environ.get('DEEPSEEK_API_KEY'))}")

print()
print("=" * 72)
print("2. 依赖可用性（CI 里刻意不装重依赖，看哪些会跳过）")
print("=" * 72)
DEPENDENCIES = [
    ("pytest", "测试框架"),
    ("pytest_cov", "覆盖率插件"),
    ("pytest_mock", "打桩插件"),
    ("langchain_core", "LangChain 核心"),
    ("langchain_text_splitters", "文本分片"),
    ("langchain_community", "Chroma / 加载器"),
    ("langchain_openai", "大模型客户端"),
    ("pypdf", "PDF 解析后端"),
    ("pydantic", "数据校验"),
    ("dotenv", "环境变量"),
    ("chromadb", "向量库（CI 不装）"),
    ("torch", "深度学习（CI 不装）"),
    ("sentence_transformers", "嵌入模型（CI 不装）"),
    ("docx", "Word 解析（可选）"),
    ("ruff", "静态检查（仅 lint job 装）"),
]
missing = []
for mod, desc in DEPENDENCIES:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  OK    {mod:<26} {ver:<12} {desc}")
    except Exception as exc:
        missing.append(mod)
        print(f"  MISS  {mod:<26} {'':<12} {desc}  ({type(exc).__name__})")

print()
print("=" * 72)
print("3. 项目模块导入（能否在最小依赖下 import）")
print("=" * 72)
sys.path.insert(0, str(PROJECT_ROOT))
import_ok = True
for mod in ["config", "src.logger", "src.loader", "src.splitter", "src.vectorstore", "src.qa"]:
    try:
        importlib.import_module(mod)
        print(f"  OK    {mod}")
    except Exception as exc:
        import_ok = False
        print(f"  FAIL  {mod}  -> {type(exc).__name__}: {str(exc)[:160]}")

print()
print("=" * 72)
print("4. 临时目录可写性（conftest 的 tmp_dir 依赖它）")
print("=" * 72)
for label, base in [
    ("系统临时目录", Path(tempfile.gettempdir())),
    ("项目内 .tmp_tests", PROJECT_ROOT / ".tmp_tests"),
]:
    probe = base / "probe_中文名"
    try:
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "中文文件名.txt").write_text("ok", encoding="utf-8")
        assert (probe / "中文文件名.txt").read_text(encoding="utf-8") == "ok"
        print(f"  OK    {label}: {base}")
        shutil.rmtree(probe, ignore_errors=True)
    except Exception as exc:
        print(f"  FAIL  {label}: {base}  -> {type(exc).__name__}: {str(exc)[:120]}")

print()
print("=" * 72)
print("5. pytest 收集（这一步能暴露 collection error）")
print("=" * 72)
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "--no-header"],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
)
tail = (result.stdout or "").strip().splitlines()
print(f"  退出码: {result.returncode}")
for line in tail[-6:]:
    print(f"  {line}")
if result.returncode != 0:
    print("  --- stderr ---")
    for line in (result.stderr or "").strip().splitlines()[-25:]:
        print(f"  {line}")

print()
print("=" * 72)
print("6. 结论")
print("=" * 72)
if missing:
    print(f"  缺失依赖: {', '.join(missing)}")
    print("  （这是 CI 的刻意取舍，对应测试会 skip；若某项导致 FAIL 才是问题）")
if not import_ok:
    print("  !! 有模块无法导入 —— 这会让相关测试整体 collection error")
    sys.exit(1)
print("  模块导入正常，继续看第 5 节的 pytest 收集结果")
sys.exit(result.returncode)
