# AgeMem HotpotQA 配置

本目录包含 AgeMem 在 HotpotQA 数据集上进行训练与评估的配置文件模板。

## 配置文件说明

| 文件 | 用途 | Workflow 注册名 |
|------|------|-----------------|
| `agemem_train.yaml` | 三阶段 GRPO 训练 | `AgeMem_hotpot_workflow_training` |
| `agemem_eval.yaml`  | Bench 模式评估   | `AgeMem_hotpot_workflow_evaluation` |

## 快速开始

### 1. 设置环境变量

```bash
export TRINITY_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export TRINITY_CHECKPOINT_ROOT_DIR=/path/to/checkpoints
export HOTPOTQA_PATH=/path/to/dataset/hotpot_qa/fullwiki
export DASHSCOPE_API_KEY=your_dashscope_key   # LLM-as-Judge / DistractorGenerator 必需
```

### 2. 修改 YAML 中的路径

若不使用环境变量，手动替换以下字段：

| 字段 | 说明 |
|------|------|
| `buffer.explorer_input.taskset.path` | HotpotQA 数据根目录 |
| `buffer.explorer_input.eval_tasksets[].path` | 评估数据路径（仅 eval） |
| `model.model_path` | 基座模型路径 |
| `model.lora_configs[].path` | LoRA checkpoint 路径（eval 时指向已训练 LoRA） |

### 3. 运行

**训练：**

```bash
ray start --head
trinity run --config examples/agemem_hotpotqa/agemem_train.yaml
```

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
