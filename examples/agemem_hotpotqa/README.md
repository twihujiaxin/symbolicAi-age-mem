# AgeMem HotpotQA 配置

本目录包含 AgeMem 在 HotpotQA 数据集上进行训练与评估的配置文件模板。

## 配置文件说明

| 文件 | 用途 | Workflow 注册名 |
|------|------|-----------------|
| `agemem_train.yaml` | E2 兼容性 heuristic dense reward 模板 | `AgeMem_hotpot_workflow_training` |
| `agemem_e0_frozen_eval.yaml` | M8b 固定 2 条 held-out 数据的 E0 基座评测 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_dry_run.yaml` | M8b 固定 6 条数据、2-GPU、单次更新 E1 smoke | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_repeat.yaml` | 同一 6 条数据的 E1 terminal-only 多 seed 训练；job 名由 `AGEMEM_E1_JOB_NAME` 注入 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_scale.yaml` | 1.5B 正式扩大：24 条 train、8 step、无 nudge；由 `agemem_e1_scale_select.py` 生成 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_scale_eval.yaml` | 扩大 run 的新进程 held-out 评测 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_repeat_eval.yaml` | 单个 repeat seed 的新进程 held-out 评测 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_stage3_answer_probe.yaml` | 同一 6 条 train 样本、T=0、最后一轮强制要求 `<answer>` 的冻结评测；不训练 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_checkpoint_eval.yaml` | M8b 新进程加载 E1 checkpoint 后的固定评测 | `AgeMem_hotpot_workflow_training` |
| `agemem_e0_4b_frozen_eval.yaml` | 独立 4B E0：同一 6+2 行、无 nudge | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_4b_dry_run.yaml` | 独立 4B E1：Qwen3-4B、6 条、1 step、无 nudge | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_4b_checkpoint_eval.yaml` | 4B E1 新进程 checkpoint 评测 | `AgeMem_hotpot_workflow_training` |
| `agemem_e1_4b_stage3_answer_probe.yaml` | 独立 4B Stage 3 `<answer>` probe：同一 6 条 train、T=0、不训练 | `AgeMem_hotpot_workflow_training` |
| `agemem_eval.yaml`  | Bench 模式评估   | `AgeMem_hotpot_workflow_evaluation` |

## 快速开始

### 1. 设置环境变量

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export TRINITY_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
export TRINITY_CHECKPOINT_ROOT_DIR=/path/to/checkpoints
export HOTPOTQA_PATH=/path/to/dataset/hotpot_qa/fullwiki
export TRINITY_MODEL_REVISION=<完整40位模型revision>
export DASHSCOPE_API_KEY=your_dashscope_key
```

E1 的 terminal reward 和固定 distractor 不调用辅助 LLM；但当前 memory
workflow 的 embedding 以及模型主动调用的 SUMMARY/CLEAR 仍可能访问 DashScope，
因此 E1 也不能被描述为端到端离线运行。M8b smoke 已在 YAML 中冻结 provider、
embedding/chat model。每次调用会立即写入 checkpoint job 下独立的
`trajectories/auxiliary_provider_calls.jsonl`，Experience 同时保存 rollout 汇总；
两者都不含 prompt、response、header 或 key。API 不返回货币金额时保持 `null`，
再用 provider 账单对账。

### 2. 修改 YAML 中的路径

若不使用环境变量，手动替换以下字段：

| 字段 | 说明 |
|------|------|
| `buffer.explorer_input.taskset.path` | HotpotQA 数据根目录 |
| `buffer.explorer_input.eval_tasksets[].path` | 评估数据路径（仅 eval） |
| `model.model_path` | 基座模型路径 |
| `model.lora_configs[].path` | LoRA checkpoint 路径（eval 时指向已训练 LoRA） |

### 3. 运行

**E2 兼容性模板：**

```bash
ray start --head
trinity run --config examples/agemem_hotpotqa/agemem_train.yaml
```

**M8b 完整 smoke（只在组内远程 GPU 门禁通过后）：**

```bash
bash scripts/autodl_m8b_preflight.sh
bash scripts/autodl_m8b_smoke.sh
```

该配置不是正式实验：它固定 M5 manifest 的 6 条 source-train 样本，并在读取时
校验 train Dataset fingerprint、source index 顺序和 Hotpot ID；同时固定 K=2，
使用 1 张 rollout GPU + 1 张 trainer GPU，并只执行 1 个 trainer step。
第二个脚本按 E0 → E1 单次更新 → 重启 Ray → checkpoint eval → postflight 的固定
顺序执行，并验证 receipt、有限 loss/KL/reward、checkpoint shards、LoRA 变化和
不同进程的 model-version-1 重载。1.5B smoke 已在远端 `e82bf54` /
`/data/hjx/Age_mem/checkpoints-attempt-002` 通过。完整环境、模型 manifest、顺序和
停止条件见 [M8b 远程 GPU 服务器执行包](../../docs/m8b_autodl_preflight.md)。

**E1 terminal-only 多 seed 重复（不进入 E3，不改冻结 dry-run YAML）：**

```bash
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints-e1-repeat
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
bash -n scripts/agemem_e1_repeat.sh
bash scripts/agemem_e1_repeat.sh
```

该脚本使用 `agemem_e1_repeat.yaml` / `agemem_e1_repeat_eval.yaml`，同一 6 条
M5 train 样本与 2 条 held-out 评测，seeds `7/17/27`，job 名为
`agemem-e1-terminal-only-repeat-s{seed}`。必须使用不含 M8b smoke job 的新
checkpoint 根目录。不要调用 `autodl_m8b_smoke.sh` 做这件事。

**Stage 3 是否写出 `<answer>`（冻结 1.5B，不训练）：**

```bash
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints-e1-answer-probe
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
bash -n scripts/agemem_e1_stage3_answer_probe.sh
bash scripts/agemem_e1_stage3_answer_probe.sh
```

保持 `stage3_max_rounds: 2`，最后一轮要求 `<answer>`；若仍无标签再追加一轮只修标签。
默认 E1/smoke YAML 不启用这些开关。不要把这次 probe 说成 DFA 或正式 E1。

**1.5B terminal-only 正式扩大（无 nudge，24 条 / 8 step）：**

```bash
export HOTPOTQA_PATH=/data/hjx/Age_mem/data/hotpot_qa/fullwiki
python scripts/agemem_e1_scale_select.py --write-yaml
# 提交生成的 configs/e1_scale.json 与 YAML 后再训练
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints-e1-scale
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
bash scripts/agemem_e1_scale.sh
```

不要改冻结 dry-run YAML，不要打开 `stage3_require_final_answer`。

**4B terminal-only E1（无 nudge，独立锁，Qwen3-4B）：**

```bash
export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen3-4B
export TRINITY_MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints-e1-4b
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
bash scripts/agemem_e1_4b.sh
```

必须先按 `docs/m8b_autodl_preflight.md` 下载冻结 revision 并生成
`.agemem_model_manifest.json`。不要复用 1.5B checkpoint 根，不要改冻结 dry-run YAML。
4B 无 nudge E1 已在 `checkpoints-e1-4b-006` 关闭（reward/F1 全 0）；不要复用该根目录。

**4B Stage 3 `<answer>` probe（不训练，不并入基线）：**

```bash
export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen3-4B
export TRINITY_MODEL_REVISION=1cfa9a7208912126459214e8b04321603b3df60c
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints-e1-4b-answer-probe
mkdir -p "$TRINITY_CHECKPOINT_ROOT_DIR"
bash -n scripts/agemem_e1_4b_stage3_answer_probe.sh
bash scripts/agemem_e1_4b_stage3_answer_probe.sh
```

不要复用 `checkpoints-e1-4b-006` 或 1.5B probe 根目录。不要把这次 probe 说成 E3 或 4B E1 基线。

**单阶段调试命令（不替代完整 smoke 脚本）：**

```bash
trinity run --config examples/agemem_hotpotqa/agemem_e0_frozen_eval.yaml
trinity run --config examples/agemem_hotpotqa/agemem_e1_checkpoint_eval.yaml
```

两者只读取 M5 manifest 固定的 2 条 held-out validation 样本。checkpoint 评测必须
在 E1 保存完成并重启 Ray/Python 进程后执行。

**评估：**

```bash
trinity run --config examples/agemem_hotpotqa/agemem_eval.yaml
```

## 训练工具轨迹

训练配置默认开启六种工具的结构化轨迹。文件位于：

```text
<checkpoint_job_dir>/trajectories/tool_calls.jsonl
```

轨迹当前为 `schema_version=2`。每次调用至少写两条 JSONL 记录：
`phase=start` 和 `phase=finish`；如果一次 Retrieve 的结果确实进入了下一次模型
输入，还会追加一条 `phase=usage`。这些记录用同一个 `call_id` 关联，每一行另有
唯一 `record_id`；同时记录
`batch_id/task_id/run_id/execution_id/stage/round/step/turn/tool_index`、参数、结果、
耗时以及调用前后的 STM/LTM 计数摘要。记录器会按敏感键名和常见文本格式尽力
脱敏，且不会写入客户端、环境变量或 embedding；但工具参数和结果本身可能包含
训练语料或其他敏感信息，因此应把整个轨迹视为敏感数据，目录已加入 `.gitignore`。
超过 `tool_trace_max_string_chars` 的字符串只保存预览、长度和 SHA-256，所以该文件
用于审计与分析，不保证能确定性重放全部工具调用。

如需指定其他位置，可设置 `AGEMEM_TOOL_TRACE_PATH`（可以是 `.jsonl` 文件，也可以
是目录）；如需在控制台同步看到每次调用的简要状态，将
`tool_trace_console` 改为 `true`。共享写入器异常或写入超过
`tool_trace_ray_timeout_seconds` 时，记录器会在主文件旁生成带主机名和进程号的
`*.fallback.jsonl`。合并主文件和回退文件时应按 `record_id` 去重，因为 Ray 的
取消是尽力而为，极端情况下同一行可能同时出现在两处。如果主文件和回退文件都
无法写入，Experience 信息中的 `tool_trace_dropped_record_count` 和
`tool_trace_last_write_error` 会显式报告记录器启动以来的累计丢失。普通写入异常
不会中断训练，但操作系统或网络文件系统本身卡死仍可能阻塞 I/O；Windows 的
`chmod(0600)` 也不等同于
NTFS 的 owner-only ACL，生产环境仍应配置目录访问控制。

轨迹会按模型输出顺序记录所有实际调用（包括内容完全相同的重复调用），但中间轮
Experience 仍沿用原训练筛选：
Stage 1 只保留 Add/Retrieve/Update，Stage 2 只保留 Summary/Clear，
Stage 3 只保留 Summary/Clear/Retrieve。其他工具可以执行并进入轨迹，但不会因此
单独产生 Experience。

`tool_reward_stats_source` 默认为 `legacy`，用于兼容旧配置按最终 STM 文本统计奖励
的口径；当前训练模板显式设为 `trace`，使用三个 Stage 中实际产生效果的完成事件
统计工具的存在性和记忆奖励；无效果、失败及格式错误的调用仍进入 attempt/轨迹
统计，并计入工具调用密度成本。工具密度按三个 Stage 的实际模型轮数计算，
Stage 3 耗尽惩罚仍单独按 Stage 3 轮数判断。这样轨迹中没有独立 Experience 的
工具仍可参与奖励，但不会因为分子、分母范围不一致而被误罚。

## 关键 workflow_args 说明

| 参数 | 说明 |
|------|------|
| `stage2_distractor_messages` | Stage 2 干扰消息条数 |
| `stage2_distractor_source` | `fixed` / `task` / `provider`；E1 禁止 provider |
| `reward_profile` | `terminal_only`（E1）或 `agemem_heuristic`（E2） |
| `terminal_reward_metric` | E1 的确定性 HotpotQA terminal metric |
| `milestone_reward_enabled` | M8a 必须为 `false` |
| `stage1_max_rounds` | Stage 1 最大多轮次数 |
| `stage2_max_rounds` | Stage 2 最大多轮次数 |
| `stage3_max_rounds` | Stage 3 最大多轮次数 |
| `max_context_tokens` | 上下文 token 上限（触发自动摘要） |
| `tool_trace_enabled` | 是否写入六工具 JSONL 轨迹 |
| `tool_trace_console` | 是否在训练日志中打印每次调用的简要状态 |
| `tool_trace_max_string_chars` | 单个字符串字段写入轨迹前的最大字符数 |
| `tool_trace_ray_timeout_seconds` | 等待共享 Ray 写入器的超时秒数，超时后切换到进程回退文件 |
| `tool_reward_stats_source` | 奖励统计来源：兼容旧实验的 `legacy` 或精确事件 `trace` |
| `use_context_tools` | 是否启用 Summary/Clear/Retrieve 工具（仅 eval 可关闭） |
| `enable_stage2_in_eval` | 评估时是否执行 Stage 2 干扰注入 |

详细说明见 [docs/AgeMem_README.md](../../docs/AgeMem_README.md)。
