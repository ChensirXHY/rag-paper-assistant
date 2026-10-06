# ============================================================
# dlenv_project 提交脚本
#
# 用法（在 PowerShell 里执行）：
#   cd E:\project\dlenv_project
#   .\commit.ps1
#
# 本次改造把 103 行的单文件脚本重构为模块化项目，修复了三个真实缺陷，
# 并补上测试、评测与 CI。提交信息按 Conventional Commits 规范分四次提交，
# 让历史能看出改造脉络（替代原来那条"提交了rag本地文献系统代码和测试代码"）。
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "=== 提交 1/4：依赖锁定与工程配置 ===" -ForegroundColor Cyan
git add requirements.txt requirements-dev.txt .env.example ruff.toml pyproject.toml .gitignore
git commit -m @"
chore: 锁定依赖版本并补充工程配置

- requirements.txt：按实际开发环境锁定版本（langchain 1.3.11 /
  chromadb 1.5.9 / torch 2.7.1+cu118），保证他人可复现
- requirements-dev.txt：补 ruff / pytest / pytest-cov / pytest-mock / pre-commit
- .env.example：环境变量模板，注明各参数的调参依据
- ruff.toml：静态检查配置，含中文项目对 pydocstyle 的取舍说明
  （D400/D415 要求 ASCII 句点结尾，中文 docstring 用"。"，故忽略）
- pyproject.toml：pytest 配置（testpaths / integration 标记 / 覆盖率门槛）
- .gitignore：覆盖新目录结构，补充备份目录与运行产物
"@

Write-Host "`n=== 提交 2/4：核心逻辑模块化 ===" -ForegroundColor Cyan
git add config.py src/ prompts/
git commit -m @"
refactor: 将 RAG_base.py 拆分为模块化结构

原实现是 103 行顶层脚本，零函数零类，配置硬编码、print 当日志、
无异常处理，无法被导入或测试。现拆分为：

- config.py           全局配置（显式 .env 路径 + 启动校验 + 调参依据注释）
- src/loader.py       多格式加载 + 中文排版清洗 + 低质页过滤
- src/splitter.py     中文标点优先的递归分片
- src/vectorstore.py  Chroma 封装 + 文件指纹增量索引
- src/qa.py           RAG 编排（LCEL）+ 结构化 Answer/Source
- src/logger.py       统一日志（替代 print）
- prompts/qa_prompt.txt  提示词独立成文件，可 diff、可 A/B 对比

修复的缺陷（详见 DEPLOY_REPORT.md）：

1. 新增文献永远不被索引
   原用 os.path.exists(CHROMA_DB_DIR) 判断是否建索引，目录存在即跳过，
   导致 papers/ 里 6 篇论文只有 2 篇被索引（152 块），另外 4 篇静默丢失。
   现改用 size+mtime 文件指纹做增量比对，覆盖 6/6 篇（429 块）。

2. .env 读取静默失败
   load_dotenv() 不带路径时从调用文件位置向上查找，从别的目录启动会
   返回 False 且不报错，API Key 变成 None 直到调用大模型才报错。
   现用 Path(__file__) 显式锚定项目根目录，并区分失败原因给出修复建议。

3. PDF 解析噪声污染嵌入向量
   中文标点前后被插入空格、图表坐标轴被解析成控制字符，实测这些噪声
   会让纯图表碎片排到最相关位置。现增加 normalize_cjk_text() 与
   is_meaningful()：字符数 -7.6%、空格数 -34%，检索结果不再含控制字符。

其余修正：
- 分片分隔符表补入中文句末标点。实测中文配置下 10/10 个块由完整句子组成，
  英文默认配置仅 0/10、残缺片段占 37%
- 向量库改用余弦距离（hnsw:space=cosine），与 BGE 归一化向量配套
- 嵌入设备由注释兜底改为 resolve_device() 自动探测
- Word 支持改为可选依赖降级，未装 python-docx 时不影响主流程
- top_k 默认值由 4 调整为 8，依据消融实验：
  Recall@1=63.6% / @4=72.7% / @8=100% / @12=100%
"@

Write-Host "`n=== 提交 3/4：CLI 入口 ===" -ForegroundColor Cyan
git add app.py
git commit -m @"
feat(cli): 重写命令行入口

支持四种模式：交互问答、单次提问、流式输出、索引状态查询。
用 rich 渲染答案面板与出处表格，索引状态可直观看出哪些文献被覆盖
（用于诊断"新增论文查不到"这类问题）。
"@

Write-Host "`n=== 提交 4/4：测试、评测与 CI ===" -ForegroundColor Cyan
git add tests/ eval/ .github/ .pre-commit-config.yaml README.md DEPLOY_REPORT.md
git commit -m @"
test: 补充测试套件、评测框架与 CI

测试（132 通过 / 3 跳过，覆盖率 82%）：
- tests/test_loader.py       清洗、过滤、注册表、指纹、加载错误提示
- tests/test_splitter.py     中文分隔符设计守护、句子完整性
- tests/test_vectorstore.py  增量索引四个分支（新增/未变/变更/删除）
- tests/test_qa.py           上下文拼装、出处去重、参数校验
- 集成测试用 --run-integration 开启（真实 PDF + 真实 Chroma，135 通过）

评测框架（eval/run_eval.py，15 题评测集）实测结果：
- 检索命中率 Recall@8 = 100%，出处准确率 80.0%
- 拒答准确率 100%，幻觉率 0%，整体通过率 93.3%
- 拒答判定用三重规则（关键词 + 长度上限 + 转折词黑名单），
  避免模型"先拒答再编"被误判为成功拒答而让幻觉率虚低

CI（.github/workflows/ci.yml）：
- lint job 跑 ruff check + ruff format --check
- test job 只装轻量依赖（不含 torch/chromadb），用假 Key 跑测试
- 覆盖率门槛 50%（CI 跳过集成测试时的保守值，本地全量为 82%）

其它：
- .pre-commit-config.yaml 装 Git 钩子，拦截大文件与私钥
- README.md 补充实测指标、消融表、技术方案与排查表
- DEPLOY_REPORT.md 记录缺陷分析、验证记录与回滚方式
"@

Write-Host "`n=== 提交完成 ===" -ForegroundColor Green
git log --oneline -6
Write-Host "`n远程状态与推送：" -ForegroundColor Yellow
git status -sb | Select-Object -First 1
Write-Host "  git push origin main"
