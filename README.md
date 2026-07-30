# 	Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents

## Table of Contents

- [1. Overview](#1-overview)
- [2. Installation](#2-installation)
- [3. Project Structure](#3-project-structure)
- [4. Data Preparation](#4-data-preparation)
- [5. Model Preparation](#5-model-preparation)
- [6. Configuration](#6-configuration)
- [7. Training](#7-training)
- [8. Evaluation](#8-evaluation)
- [9. Standalone AgentScope Example](#9-standalone-agentscope-example)

---

## 1. Overview

AgeMem is built on Trinity-RFT and performs reinforcement fine-tuning (RFT) on HotpotQA to train LLM agents with **context management** and **long-term memory management** capabilities.

The model uses six callable tools:

| Tool | Type | Function |
|------|------|----------|
| `Summary_context` | STM (context) | Compresses historical dialogue to save tokens |
| `Clear_context` (`Filter_context`) | STM (context) | Removes irrelevant context by semantic criteria |
| `Retrieve_memory` | STM (context) | Retrieves relevant long-term memory into current context |
| `Add_memory` | LTM | Adds new memory to vector store |
| `Update_memory` | LTM | Updates existing memory |
| `Delete_memory` | LTM | Deletes memory by ID |

### Three-stage training pipeline

```text
Stage 1: Casual interaction  - Learn Add/Update/Delete memory behavior from context facts
Stage 2: Distractor injection - Learn Clear/Summary behavior under noisy context
Stage 3: Formal QA           - Learn integrated retrieval + reasoning + context control
```

---

## 2. Installation

### 2.1 Clone the repo

```bash
git clone https://github.com/y1y5/AgeMem
cd AgeMem
```

### 2.2 Create a virtual environment

```bash
# Conda (recommended)
conda create -n trinity python=3.10.19
conda activate trinity

# Or venv
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2.3 Install Trinity-RFT

```bash
# Editable install (recommended)
pip install -e ".[dev]"

# Optional: flash-attn acceleration
pip install -e ".[flash_attn]"
# If build fails, try:
# pip install flash-attn==2.8.1 --no-build-isolation
```

### 2.4 Set environment variables

```bash
# Base model path
export TRINITY_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

# Checkpoint root
export TRINITY_CHECKPOINT_ROOT_DIR=/path/to/checkpoints

# HotpotQA fullwiki path
export HOTPOTQA_PATH=/path/to/dataset/hotpot_qa/fullwiki

# DashScope API key (required for distractor generation and LLM-as-judge)
export DASHSCOPE_API_KEY=your_dashscope_api_key

# Tokenizer path (optional, defaults to bert-base-uncased)
export TOKENIZER_PATH=/path/to/bert-base-uncased

# WandB API key (optional)
export WANDB_API_KEY=your_wandb_api_key
```

---

## 3. Project Structure

```text
AgeMem/
├── trinity/common/workflows/
│   ├── memory_context/
│   │   ├── train_hotpotQA.py
│   │   ├── eval_hotpotQA.py
│   │   ├── utils.py
│   │   ├── memory_store.py
│   │   ├── workflow_prompt.py
│   │   └── workflow_metrics.py
│   └── memory_reward/
│       └── my_reward.py
├── examples/
│   └── agemem_hotpotqa/
│       ├── agemem_train.yaml
│       ├── agemem_eval.yaml
│       └── README.md
├── AgeMem_code_agentscope/
├── docs/
│   └── AgeMem_README.md
└── pyproject.toml
```

---

## 4. Data Preparation

AgeMem uses [HotpotQA](https://hotpotqa.github.io/) in fullwiki format.

Expected directory layout:

```text
/path/to/dataset/hotpot_qa/
├── distractor/
├── fullwiki/
└── ...
```

### 4.1 Required fields

| Field | Type | Description |
|------|------|-------------|
| `question` | `str` | Input question |
| `answer` | `str` | Ground-truth answer (can be missing in some test sets) |
| `context` | `dict` | `{"title": [...], "sentences": [[...], ...]}` |
| `supporting_facts` | `dict` (optional) | `{"title": [...], "sent_id": [...]}` |

### 4.2 Dataset path in YAML

```yaml
buffer:
  explorer_input:
    taskset:
      storage_type: file
      path: '/path/to/dataset/hotpot_qa/fullwiki'
      split: 'train'
      format:
        prompt_key: 'question'
        response_key: 'answer'
```

---

## 5. Model Preparation

### 5.1 Download base model

```bash
# HuggingFace
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /path/to/model/Qwen2.5-7B-Instruct

# Or ModelScope
modelscope download Qwen/Qwen2.5-7B-Instruct \
  --local_dir /path/to/model/Qwen2.5-7B-Instruct
```

### 5.2 Set model path in YAML

```yaml
model:
  model_path: ${oc.env:TRINITY_MODEL_PATH,/path/to/Qwen2.5-7B-Instruct}
```

---

## 6. Configuration

### 6.1 Training config (`agemem_train.yaml`)

Key fields:

| Field | Description |
|------|-------------|
| `buffer.explorer_input.taskset.path` | HotpotQA training set path |
| `buffer.explorer_input.default_workflow_type` | `AgeMem_hotpot_workflow_training` |
| `algorithm.algorithm_type` | `grpo` |
| `algorithm.repeat_times` | Rollouts per sample (default 8) |
| `workflow_args.stage2_distractor_messages` | Stage 2 distractor count |
| `workflow_args.stage3_max_rounds` | Stage 3 max rounds |
| `workflow_args.max_context_tokens` | Context token budget |

### 6.2 Evaluation config (`agemem_eval.yaml`)

Key fields:

| Field | Description |
|------|-------------|
| `mode` | `bench` (evaluation mode) |
| `buffer.explorer_input.default_workflow_type` | `AgeMem_hotpot_workflow_evaluation` |
| `buffer.explorer_input.eval_tasksets` | Evaluation tasksets |
| `explorer.bench_on_latest_checkpoint` | Evaluate latest checkpoint or not |
| `explorer.eval_on_startup` | Run evaluation on startup |
| `explorer.env_vars.DASHSCOPE_API_KEY` | API key for LLM judge |
| `workflow_args.use_context_tools` | Enable Summary/Clear/Retrieve |
| `workflow_args.enable_stage2_in_eval` | Enable Stage 2 distractors in eval |

---

## 7. Training

### 7.1 Start Ray cluster

```bash
# Single machine
ray start --head

# Worker node
ray start --address=<master_ip>:6379
```

### 7.2 Run training

```bash
trinity run --config examples/agemem_hotpotqa/agemem_train.yaml
```

Training loop:

1. Explorer runs `AgeMem_hotpot_workflow_training` for three-stage rollouts
2. Experiences are written into buffer
3. Trainer updates policy with GRPO
4. Checkpoints are synchronized by configured interval

### 7.3 Resume from checkpoint

```yaml
continue_from_checkpoint: true
```

Make sure `checkpoint_root_dir` and experiment `name` match the original run.

### 7.4 Monitoring (optional)

Enable the `monitor` section in YAML and set `WANDB_API_KEY`.

---

## 8. Evaluation

```bash
trinity run --config examples/agemem_hotpotqa/agemem_eval.yaml
```

Before running:

- Ensure `model.lora_configs[].path` points to your checkpoint
- Ensure all `eval_tasksets` paths are correct
- Ensure `DASHSCOPE_API_KEY` is set

---

## 9. Standalone AgentScope Example

`AgeMem_code_agentscope/` provides a standalone demo that does not depend on the Trinity-RFT training pipeline.

```bash
pip install -r AgeMem_code_agentscope/requirements.txt
export DASHSCOPE_API_KEY=your_key
python -m AgeMem_code_agentscope.main
```

See `AgeMem_code_agentscope/README.md` for details.

---

## Acknowledgement

This project is built on top of [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT), an excellent open-source reinforcement fine-tuning framework for LLM agents. We sincerely thank the Trinity-RFT team for their outstanding contribution to the community.

## Citation

If this codebase helps your research, please cite the AgeMem paper.

```bibtex
@article{yu2026agentic,
  title={Agentic memory: Learning unified long-term and short-term memory management for large language model agents},
  author={Yu, Yi and Yao, Liuyi and Xie, Yuexiang and Tan, Qingquan and Feng, Jiaqi and Li, Yaliang and Wu, Libing},
  journal={arXiv preprint arXiv:2601.01885},
  year={2026}
}
```
