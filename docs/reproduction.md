# M0 上游与 standalone 复现记录

## 范围

本记录只覆盖 `PROJECT_HANDOFF.md` 中的 M0：已有仓库接管、运行环境检查、
standalone AgentScope demo 和现有记忆工具 smoke test。未修改 GRPO、训练工作流、
数据库、LTL 或自动机实现。

检查日期：2026-08-02（Asia/Shanghai）

## 仓库状态

- 工作目录：`D:\Project\Age-Mem\AgeMem`
- Git 分支：`main`，跟踪 `origin/main`
- 当前提交：`c82a054726b7a18ed191f52aa7ab71add2d8a283`
- 提交标题：`Initial import of symbolic AgeMem project`
- 远端：`https://github.com/twihujiaxin/symbolicAi-age-mem.git`
- 当前仓库只保留了上述项目远端，没有记录 AgeMem 和 Trinity-RFT 各自的精确上游
  commit；因此不能从本地历史可靠恢复这两个上游 SHA。
- `pyproject.toml` 声明的 Trinity-RFT 项目版本为 `0.3.1`。

接管时已有下列用户修改，本次 M0 不修改、不清理也不覆盖它们：

```text
M  .gitignore
M  examples/agemem_hotpotqa/README.md
M  examples/agemem_hotpotqa/agemem_train.yaml
M  trinity/common/workflows/memory_context/memory_store.py
M  trinity/common/workflows/memory_context/train_hotpotQA.py
M  trinity/common/workflows/memory_context/utils.py
M  trinity/common/workflows/memory_reward/my_reward.py
?? tests/common/tool_trace_test.py
?? trinity/common/tool_trace.py
```

主要被忽略内容包括 `.venv/`、`.vscode/`、`.ruff_cache/`、`tmp/`、各级
`__pycache__/` 和本地 `config`。`config` 被视为本地凭据文件，本次未读取其内容。

工作区有大量未提交实现，暂不建议切换或新建开发分支。应先由用户决定如何保存
这些修改，再开始 M1。

## 环境

| 项目 | 检查结果 |
|---|---|
| 操作系统 | Windows NT 10.0.26200.0 |
| PowerShell | 5.1.26100.8875 |
| 系统 `py` 默认版本 | Python 3.14.4，不满足项目 `<3.13` 约束，不使用 |
| 复用环境 | `.venv`（Conda prefix） |
| `.venv` Python | 3.10.20 |
| AgentScope | 1.0.21 |
| MCP | 1.27.2 |
| OpenAI SDK | 2.41.0 |
| Pydantic | 2.13.4 |
| shortuuid | 1.0.13 |
| Ollama Python 包 | 未安装；本地 Ollama 后端不可用 |
| PyTorch | 未安装；M0 standalone 不需要 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB |
| NVIDIA 驱动 | 591.94 |

`pip check` 返回 `No broken requirements found.`。standalone 包、AgentScope 和本地
`trinity` 均可以从仓库及 `.venv` 正常导入。M0 全程显式使用：

```powershell
.\.venv\python.exe
```

没有创建新环境，也没有安装或升级依赖。

## standalone 入口

本地 README 声明的入口为：

```powershell
Set-Location 'D:\Project\Age-Mem\AgeMem'
$env:DASHSCOPE_API_KEY = Read-Host 'Enter DASHSCOPE_API_KEY locally'
.\.venv\python.exe -m AgeMem_code_agentscope.main
```

真实 standalone 复现已由用户在本地完成并于 2026-08-02 明确确认。当前 Codex
子进程没有继承 `DASHSCOPE_API_KEY`，因此本次接管没有重复发送付费 API 请求，
也没有读取本地 `config`。该真实运行结果属于用户确认，而不是当前 Codex 进程的
独立网络复测。

README 还提供 Ollama 主模型模式，但当前机器未发现 `ollama` 命令，且现有 demo
即使使用 Ollama 主模型，embedding、摘要和相似度判断仍需要 DashScope。因此它
不能作为完全离线的替代入口。

## 记忆工具 smoke test

为避免网络和模型选择的不确定性，使用内存中的确定性 embedding 和假辅助聊天
客户端直接实例化同一个 `AgeMem` Agent，并对工具结果、memory ID、正文和删除后
状态执行断言。没有调用真实 LLM，也没有把测试凭据写入文件。

验证顺序：

```text
ADD("Project codename is Atlas.")
  -> RETRIEVE("Project codename is Atlas.")
  -> UPDATE(memory_id, "Project codename is Borealis.")
  -> RETRIEVE("Project codename is Borealis.")
  -> DELETE(memory_id, confirmation=True)
```

结果：

```text
registered_tools=add_memory,delete_memory,filter_context,retrieve_memory,summary_context,update_memory
sequence=ADD>RETRIEVE>UPDATE>RETRIEVE>DELETE
memory_id_created=True
post_delete_count=0
M0_OFFLINE_TOOL_SMOKE=PASS
```

这验证了六工具注册以及 ADD、RETRIEVE、UPDATE、DELETE 的确定性状态链路；真实
模型是否自主选择正确工具仍取决于实际模型响应，由用户已完成的 standalone 运行
提供外部确认。

## 已知问题

1. `AgeMem.__init__` 使用
   `kwargs.get("chat_client", chat_client())`。Python 会先求值默认参数，所以即使
   调用方显式传入假客户端，仍会构造 DashScope 客户端；无凭据的第一次离线实例化
   因此失败。smoke test 只在子进程中使用明显的无效占位值绕过客户端构造，所有
   实际网络方法均已替换，随后测试通过。M0 按约束没有修改该核心代码。
2. `.venv` 未安装 `ollama` 包，本机也未发现 Ollama 命令；这不影响默认 DashScope
   路径，但意味着 README 中的本地主模型选项目前不可用。
3. 当前仓库无法给出 AgeMem 与 Trinity-RFT 的独立上游 commit，只能记录合并仓库
   commit 和 `pyproject.toml` 版本。
4. 工作区已有大规模未提交修改；进入 M1 前应先决定提交、分支或其他保存策略。

## 本次执行的只读/验证命令类别

- `git status --short --branch`
- `git remote -v`
- `git log -1`
- `git diff --stat` / `git diff --numstat`
- `git ls-files --others --ignored --exclude-standard --directory`
- `.\.venv\python.exe --version`
- `.\.venv\python.exe -m pip check`
- Python 包版本和导入路径检查
- 不写文件的 inline Python 记忆工具 smoke test

## 新终端复现清单

1. 打开仓库根目录。
2. 确认 `git status --short --branch`，不要清理用户修改。
3. 运行 `.\.venv\python.exe -m pip check`。
4. 只在本地终端设置 `DASHSCOPE_API_KEY`，不要写入仓库或命令记录。
5. 运行 `.\.venv\python.exe -m AgeMem_code_agentscope.main`。
6. 开启 `AGEMEM_SHOW_TOOL_TRACE=1` 后，依次要求 Agent 添加、检索、更新、再次检索
   和删除一条测试记忆，并核对输出中的 memory ID。
7. 退出后再次运行 `git status`，确认没有凭据或意外文件进入版本控制。
