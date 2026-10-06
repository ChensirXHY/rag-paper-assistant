# dlenv_project 重构部署报告

> 部署时间：2026-10-06
> 仓库位置：`E:\project\dlenv_project`
> 环境：conda `dl_env`（Python 3.10.20）｜ langchain 1.3.11 ｜ torch 2.7.1+cu118 ｜ RTX 3060

---

## 一、本次修复的三个真实缺陷

这三个都不是"代码风格问题"，是会**实际损害检索效果**或**让程序假死**的缺陷。

### 缺陷 1：4 篇文献从未被索引（最严重）

**现象**：`papers/` 里有 6 篇论文、共 446 个文本块，但向量库里只有 **152 条**，
仅覆盖 2 篇论文。另外 4 篇论文**永远不会被检索到**，而且没有任何报错。

**根因**（原 `RAG_base.py` 第 51 行）：

```python
if not os.path.exists(CHROMA_DB_DIR):   # ← 目录存在就永远跳过重建
    vectorstore = Chroma.from_documents(...)
else:
    vectorstore = Chroma(persist_directory=..., ...)
```

只要 `chroma_db/` 目录存在（哪怕里面是空的、过时的、只索引了一部分的），
程序就永远走 `else` 分支复用旧库。往 `papers/` 里加新论文 → 不会被嵌入 → 提问永远查不到。

**修复**：改用 `size + mtime` 文件指纹做增量比对（`src/vectorstore.py`）。
每次启动对比"库中已索引文件的指纹"与"磁盘现状"，只嵌入新增/变更的文件，
并清理已删除文件的残留向量。

**验证结果**：

```
修复前：152 个块，覆盖 2/6 个文件
修复后：429 个块，覆盖 6/6 个文件
幂等性：第二次运行报告"已是最新"，块数不变
```

---

### 缺陷 2：API Key 读取会静默失败

**现象**：把脚本从别的目录启动、或在别的项目里 import 时，
`DEEPSEEK_API_KEY` 读出来是 `None`，程序继续跑，直到调用大模型时才报一个
难以定位的错误。原代码只写了 `load_dotenv()`，没有检查返回值。

**根因**：`load_dotenv()` 不带参数时，**从调用它的文件所在目录向上查找** `.env`，
而不是从当前工作目录查找。实测在本项目上它返回 `False`（加载失败）且不报错。

**修复**（`config.py`）：

```python
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
_ENV_LOADED = load_dotenv(ENV_FILE)      # 显式锚定路径，返回 False 时给出明确提示
```

`validate()` 会区分两种失败原因并给出对应修复建议：
"`.env` 文件没找到" vs "`.env` 找到了但没有 `DEEPSEEK_API_KEY`"。

---

### 缺陷 3：PDF 解析噪声污染嵌入向量

**现象**：检索到的片段里出现 `proposed ， including`、`数据集 、 特征`
（中文标点前后被插入空格），以及 `\x06\n\x04 \x04 \x07`（图表坐标轴被解析成控制字符）。
实测这些噪声甚至让一条纯图表碎片排到了"最相关片段"的位置。

**修复**（`src/loader.py`）：
- `normalize_cjk_text()`：清除控制字符、移除中文标点/汉字之间的多余空格
- `is_meaningful()`：过滤掉有效字符不足 30 个的图表碎片页

**验证结果**：

```
字符数（3 篇抽样）：97,235 → 89,849（-7.6%）
空格数（3 篇抽样）：21,249 → 13,938（-34%）
文本块数：446 → 429（过滤 17 个低质块）
检索结果含控制字符：修复前 有 → 修复后 无
```

**注意**：清洗**保留换行符**。换行是分片器的语义边界，压掉会让段落信息丢失。

---

## 二、代码结构变化

### 改造前

```
dlenv_project/
├── .env
├── .gitignore              (193 B)
├── RAG_base.py             (103 行顶层脚本，零函数零类)
├── test_llm.py             (API 连通性 demo，不是测试)
├── papers/  chroma_db/  local_models/
```

`RAG_base.py` 的问题：配置硬编码、`print` 当日志、无异常处理、
无 `if __name__ == "__main__"`、无法被 import 或测试。

### 改造后

```
dlenv_project/
├── .env                     # 密钥（已在 .gitignore）
├── .env.example             # 新增：环境变量模板
├── .gitignore               # 扩充：覆盖新目录结构
├── README.md                # 新增：含实测指标与排查表
├── app.py                   # CLI 入口（rich 美化，4 种模式）
├── config.py                # 全局配置（显式 .env 路径 + 启动校验）
├── requirements.txt         # 新增：按真实环境锁定版本
├── requirements-dev.txt     # 新增：测试与代码质量工具
├── prompts/
│   └── qa_prompt.txt        # 提示词独立成文件，可 diff / A/B 对比
├── src/
│   ├── loader.py            # 多格式加载 + 中文清洗 + 低质过滤
│   ├── splitter.py          # 中文标点优先分片
│   ├── vectorstore.py       # Chroma 封装 + 指纹增量索引
│   ├── qa.py                # RAG 编排（LCEL）+ 结构化 Answer
│   └── logger.py            # 统一日志
├── eval/                    # 评测框架（见下）
├── tests/                   # 单元测试
└── papers/  chroma_db/  local_models/
```

### 顺带修正的技术债

| 项目 | 改造前 | 改造后 | 原因 |
|---|---|---|---|
| 检索链 | `RetrievalQA.from_chain_type` | LCEL 直接组装 | `RetrievalQA` 在 LangChain 1.x 已归入 legacy 的 `langchain-classic` 包 |
| 向量库导入 | `langchain_community.vectorstores.Chroma` | 同左 | 该环境未安装 `langchain_chroma` 独立包，保持 community 导入 |
| 设备配置 | `{"device": "cuda"}` + 注释"无显卡则写 cpu" | `resolve_device()` 自动探测 | 用自动化替代注释兜底 |
| 配置读取 | 模块顶层硬编码 | `config.Settings` + 环境变量 | 换机器/换知识库无需改源码 |
| 向量库距离 | 默认 l2 | `hnsw:space=cosine` | BGE 用余弦相似度，与归一化向量配套 |
| Word 支持 | 无（简历写"多格式"但代码只支持 PDF） | 注册表 + 可选依赖降级 | 补齐简历与代码的一致性；未装 python-docx 时自动降级而非崩溃 |

---

## 三、验证记录

所有验证都在 `E:\project\_staging_test\`（仓库完整副本）和真实仓库上跑过。

| 验证项 | 结果 |
|---|---|
| `python config.py` 配置诊断 | 通过，`.env` 加载成功，API Key 读取正常 |
| `python app.py --rebuild --sync` | 通过，6 个文献 → 429 块，耗时约 60 秒（GPU） |
| 增量索引幂等性 | 通过，第二次运行报告"已是最新"，块数稳定 429 |
| 索引覆盖率自检 | 通过，6/6 文献，无"覆盖不完整"警告 |
| 检索层（无需 API Key） | 通过，6 篇文献全部可被检索命中 |
| MMR vs 相似度对比 | 通过，MMR 来源多样性 1→2~3 |
| 出处去重 | 通过，同页多块命中时正确去重 |
| 流式输出 | 通过（代码路径验证，实际输出受 API Key 影响） |
| `normalize_cjk_text` 单元验证 | 通过（含控制字符、标点空格、多空格、换行保留） |

### 未完成验证的部分（已解决）

**API Key 曾失效**，导致端到端问答无法实测。当时的报错：

```
Error code: 401 - Authentication Fails, Your api key: ****d9e7 is invalid
```

**现已解决**：你更新了 `.env`（新 Key 尾号 `24cc`，模型 `deepseek-v4-pro`），
所有端到端指标已补测完成，见下节。

### 端到端评测结果（15 题，真实 API）

| 指标 | top_k=4 | **top_k=8（默认）** |
|---|---|---|
| 检索命中率 Recall@K | 72.7% | **100%** |
| 出处准确率 | 66.7% | **80.0%** |
| 拒答准确率 | 100% | **100%** |
| 幻觉率 | 0% | **0%** |
| 整体通过率 | 73.3% | **93.3%** |
| 平均延迟 | 11.3 s | 18.2 s |
| P95 延迟 | 22.3 s | 40.5 s |

检索深度消融：`Recall@1 = 63.6% / @2 = 72.7% / @4 = 72.7% / @8 = 100% / @12 = 100%`
→ `top_k=8` 是性价比拐点，已设为默认值（`config.py` 中附了依据注释）。

### 测试与代码质量

| 项目 | 结果 |
|---|---|
| 单元测试 | 132 通过 / 3 跳过（integration 标记） |
| 集成测试 | 135 通过（真实 PDF + 真实 Chroma） |
| 覆盖率 | 82%（loader 96% / vectorstore 89% / splitter 100% / logger 100% / qa 53%） |
| ruff check | 全部通过 |
| ruff format | 全部合规 |

### 一处文档纠错（重要）

`src/splitter.py` 的 docstring 原先写"实测 6/6 分片全部对齐句末"，**这个说法是错的**。
错误来源：`keep_separator=True`（默认值）会把分隔符挂到**下一块的开头**，
而我当时的验证脚本把分隔符丢掉了，人为制造了对齐假象。

实测真实行为（`chunk_size=60`，句末标点明确的合成文本）：

| 分隔符配置 | 块尾是句末标点 | 内容全由完整句子组成的块 | 残缺片段 |
|---|---|---|---|
| 中文标点优先 | 0% | **100%** | **0%** |
| 英文默认 | 8% | 0% | 37% |

**结论不变**：中文分隔符表确实有价值，但价值在"句子不被拦腰截断"
（100% vs 0%），而不是"块尾对齐句末"。文档与测试均已按实测改写。

### 另外发现的一个不一致

`.env` 里配置的模型是 `deepseek-v4-flash`，而简历上写的是 **DeepSeek V4 Pro**。
请核对哪个是你实际在用的（可能是为了控制成本用了 flash），
简历描述应与实际一致 —— 面试官可能追问模型选型理由。

---

## 四、回滚方式

部署前做了两层备份，任何一步出问题都能还原：

| 备份目录 | 内容 | 用途 |
|---|---|---|
| `_backup_before_refactor_20261006_203404/` | 最原始的 `RAG_base.py`、`test_llm.py`、`.gitignore` | 回到重构前状态 |
| `_backup_deploy_20261006_205234/` | 部署前的代码 + **原始 chroma_db**（152 条向量） | 回到部署前状态 |

回滚示例：

```powershell
cd E:\project\dlenv_project
Remove-Item src, prompts, config.py, app.py -Recurse -Force
Copy-Item _backup_before_refactor_20261006_203404\* . -Force
Copy-Item _backup_deploy_20261006_205234\chroma_db . -Recurse -Force
```

两个备份目录已加入 `.gitignore`，不会被提交。

---

## 五、下一步

按优先级：

1. **恢复 API Key**，补齐 README 的四项质量指标（这是简历"有效回答占比提升"的依据）
2. **提交代码**，用规范的中文提交信息替代原来的单条"提交了rag本地文献系统代码和测试代码"
3. **重命名仓库** `dlenv_project` → `rag-paper-assistant`，并填上 Description / Topics
4. **跑通 CI**，把绿色徽章放进 README
5. 选修：FastAPI 接口层 + Streamlit 前端，坐实"全栈"定位
