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

## 关键 workflow_args 说明

| 参数 | 说明 |
|------|------|
| `stage2_distractor_messages` | Stage 2 干扰消息条数 |
| `stage1_max_rounds` | Stage 1 最大多轮次数 |
| `stage2_max_rounds` | Stage 2 最大多轮次数 |
| `stage3_max_rounds` | Stage 3 最大多轮次数 |
| `max_context_tokens` | 上下文 token 上限（触发自动摘要） |
| `use_context_tools` | 是否启用 Summary/Clear/Retrieve 工具（仅 eval 可关闭） |
| `enable_stage2_in_eval` | 评估时是否执行 Stage 2 干扰注入 |

详细说明见 [docs/AgeMem_README.md](../../docs/AgeMem_README.md)。
