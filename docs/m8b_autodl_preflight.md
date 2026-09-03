# M8b 远程 GPU 服务器执行包

更新时间：2026-09-02

> 部署更新：当前真实运行目标已从 AutoDL 改为组内远程 GPU 服务器，工作区根目录为
> `/data/hjx/Age_mem`，Git 仓库目录为 `/data/hjx/Age_mem/AgeMem`。现有
> `autodl_m8b_*.sh` 文件名、`configs/m8b_autodl_preflight.json` 和内部
> `--mode autodl` 作为兼容入口保留；这里的 `autodl` 表示启用严格 Linux/GPU
> fail-closed 门禁，不再表示云平台品牌。

## 当前结论

上卡前的代码、数据、配置、provider、运行时失败传播和产物验收门禁已经可执行化。
当前仍未运行真实 GPU、Ray、vLLM、模型 rollout、优化器或 checkpoint 重载，
因此 M8b 仍是“待在组内远程 GPU 服务器执行”，不是“已经通过”。

门禁默认不访问网络、不启动 Ray、不调用 LLM/embedding，也不启动训练。

## 固定输入

- E1 单次更新：`examples/agemem_hotpotqa/agemem_e1_dry_run.yaml`；
- E0 基座模型冻结评测：`examples/agemem_hotpotqa/agemem_e0_frozen_eval.yaml`；
- E1 checkpoint 新进程评测：`examples/agemem_hotpotqa/agemem_e1_checkpoint_eval.yaml`；
- 版本锁：`configs/m8b_autodl_preflight.json`；
- M5 数据清单：`data/splits/hotpotqa_smoke_manifest.json`。
- 模型清单：`$TRINITY_MODEL_PATH/.agemem_model_manifest.json`。

E1 训练只使用固定 6 条 source-train 样本、K=2、1 个 trainer step。E0 和
checkpoint 评测只使用 M5 的 2 条 held-out validation 样本。三份配置都固定
terminal-only reward、固定 Stage-2 干扰和同一个 DashScope provider profile。

## 本地检查

在 Windows 仓库根目录运行：

```powershell
.\.venv\python.exe scripts\agemem_m8b_preflight.py --mode local
```

本地模式必须通过配置、manifest、真实 fullwiki split/fingerprint/Hotpot ID 和轻量
依赖检查。没有 GPU、Ray、PyTorch、vLLM、veRL、模型目录或云端 key 会显示为
`WARN/SKIP`，不会被误记成远程 GPU 已通过。

1.5B 锁干净提交后的本地只读预检结果为 `18 PASS / 0 FAIL / 2 WARN / 11 SKIP`。
两项 WARN 是未注入云端 key 和仓库根部存在本地 ignored 凭据文件；11 项 SKIP
包含尚未配置本地 1.5B 模型路径，以及只能在远程 Linux/GPU 完整环境验证的
GPU/runtime 项。严格 runtime suite 已冻结
发现数为 `m8a=142`、`all=318`，少跑、漏跑、FAIL、ERROR 或 SKIP 都会失败。
其中 `m8a` scope 已纳入 Stage 1 storage budget、Stage 2 query-delayed challenge
和统一 anti-shortcut canary 三个模块，共 26 项测试；另有 9 项独立 stress
协议/反事实回归。这些测试不改动 E1 YAML 或
`terminal_only` reward profile。

本地 `config` 是 ignored 凭据文件，门禁会明确警告。迁移代码必须使用 Git，不得
把该文件、`.env`、`runs/`、数据库、轨迹或历史日志整体复制到云端。现有 key 应先
轮换；远程服务器只通过密钥配置或环境变量注入。

## 组内远程服务器路径与环境

当前模型目标限定为 1.5B 与 4B，不再考虑 7B。M8b 先执行锁定的 1.5B smoke；
4B 后续使用独立模型 manifest 与配置锁。使用同机两张 RTX A6000 48GB（允许
四卡宿主机显式选择两张），并把模型、数据和 checkpoint 放在持久盘：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export TRINITY_MODEL_PATH=/data/hjx/Age_mem/models/Qwen2.5-1.5B-Instruct
export HOTPOTQA_PATH=/data/hjx/Age_mem/data/hotpot_qa/fullwiki
export TRINITY_CHECKPOINT_ROOT_DIR=/data/hjx/Age_mem/checkpoints
export AGEMEM_EXPECTED_COMMIT=<本地已提交的完整40位commit>
export TRINITY_MODEL_REVISION=<Qwen模型的完整40位commit revision>
export DASHSCOPE_API_KEY=<由服务器密钥配置或隐藏输入注入>
```

不要把 key 写进 shell history、YAML、JSON、命令参数或报告。预检只记录
`DASHSCOPE_API_KEY` 是否存在，从不读取或输出其值。

按上游 README 建立 Python 3.10 环境并从固定 commit editable 安装：

```bash
conda create -n agemem-m8b python=3.10.19 -y
conda activate agemem-m8b
python -m pip install --upgrade pip
python -m pip install -e ".[m8b,dev]"
```

不要为消除单个依赖错误批量升级。版本锁会检查 Trinity `0.3.1`、veRL `0.5.0`、
Ray `>=2.48.0`、vLLM `0.9.1..0.10.2`、Transformers `4.53..4.57` 及其他关键
包；若上游安装得到不兼容组合，应停止并记录，不要临时放宽锁。

模型必须是 `Qwen/Qwen2.5-1.5B-Instruct` 的固定 40 位 revision。模型下载完成且内容
不再变化后，在模型目录旁生成一次离线 SHA-256 清单：

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$TRINITY_MODEL_REVISION" \
  --local-dir "$TRINITY_MODEL_PATH"

python scripts/agemem_m8b_model_manifest.py \
  --model-path "$TRINITY_MODEL_PATH" \
  --repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --revision "$TRINITY_MODEL_REVISION"
```

预检会核对 Qwen2.5-1.5B 结构、tokenizer/chat template、单文件
`model.safetensors`、全部物料文件清单、每个文件的大小与 SHA-256，以及总权重
不少于 3 GB。不要使用 `--force`
掩盖模型目录漂移；目录变化时应重新确认来源和 revision 后再生成清单。

随后使用冻结的 1.5B tokenizer 重跑 stress，并把证据写入持久盘而不是 Git
工作树：

```bash
stress_dir="$TRINITY_CHECKPOINT_ROOT_DIR/anti_shortcut_stress/$AGEMEM_EXPECTED_COMMIT"
python scripts/agemem_anti_shortcut_stress.py \
  --tokenizer-path "$TRINITY_MODEL_PATH" \
  --tokenizer-revision "$TRINITY_MODEL_REVISION" \
  --tokenizer-repository-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir "$stress_dir" \
  --docs-path "$stress_dir/report.md"
```

## 一键门禁

上述环境变量和目录就绪后，在仓库根目录运行：

```bash
bash -n scripts/autodl_m8b_preflight.sh
bash -n scripts/autodl_m8b_smoke.sh
bash scripts/autodl_m8b_preflight.sh
```

该脚本执行两部分：

1. 校验干净工作树、固定 commit、三份配置 digest、M5 manifest、嵌套 `.env`/
   ignored 凭据隔离、模型 provenance、持久盘剩余空间和干净 job 目录；
2. 核对 fullwiki 三个 split/fingerprint、6 条 train 与 2 条 held-out 的 Hotpot ID
   和行内容 SHA-256，并用 Trinity structured Config 解析三份 YAML；
3. 核对关键包版本；保留完整物理 GPU 清单，但只按 `CUDA_VISIBLE_DEVICES` 选择的
   两张卡执行门禁。每张总显存至少 48,000 MiB、空闲显存至少 47,000 MiB，并要求
   `nvidia-smi` 物理 UUID 与 PyTorch 重映射设备 UUID 对齐；数值选择器必须配合
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`；
4. 运行锁定的 318 项 M1～M8b/tool-trace/anti-shortcut 回归，并把数量漂移或任何
   `FAIL/ERROR/SKIP` 都视为失败。

报告保存到：

```text
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_preflight/$AGEMEM_EXPECTED_COMMIT/
  preflight_report.json
  runtime_gate_report.json
```

脚本成功时仍不会启动 Ray 或训练。

## Provider 记录口径

首轮 smoke 保留现有 DashScope provider，不同时迁移本地 embedding。配置固定：

```text
endpoint             https://dashscope.aliyuncs.com/compatible-mode/v1
embedding            text-embedding-v4, dimensions=256
SUMMARY/CLEAR chat  qwen-max
```

每一次 provider 调用都会立即追加到独立、加锁并 `fsync` 的元数据 JSONL：

```text
<checkpoint_job_dir>/trajectories/auxiliary_provider_calls.jsonl
```

记录键为 `task_id / rollout_id / execution_id / call_index`，包含 eval 标记、provider、
model、成功/失败、错误类型、延迟和 provider 返回的 token usage。失败调用和 eval
调用不依赖 Experience 持久化；SDK 自动重试已关闭，因此一条记录对应一次 HTTP
逻辑尝试。Experience 仍保存当前 rollout 汇总。两处都不保存请求正文、响应正文、
HTTP header 或 key，外围日志也只记录异常类型。OpenAI-compatible API 不返回货币
金额时，`cost.amount` 必须保持 `null`，不得伪造；实验结束后按运行时间和 usage
与 provider 账单对账。

## GPU smoke 顺序

只有一键门禁完全通过后，才由用户主动运行 GPU smoke：

```bash
bash scripts/autodl_m8b_smoke.sh
```

该脚本拒绝已有 Ray cluster、旧日志或旧 postflight 证据；随后依次执行严格预检、
E0、E1 单次更新、停止并重启 Ray、checkpoint eval 和只读 postflight。任一 CLI、
receipt、checkpoint 或 postflight 失败都会立即停止。它不会并行启动阶段，也不会
自动扩大数据、step 或 seed。

最终 postflight 硬性核对：E0 的 model-version-0 完整 held-out 指标；step 1 的有限
loss/KL/reward 和 actor-update sentinel；`trainer_meta.json`、checkpoint marker、
完整非空 model/optimizer/extra-state shards；训练后 LoRA 与 `dummy_lora` 不同；以及
checkpoint eval 的 model-version-1 receipt 来自不同的进程执行 ID。证据位于：

```text
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_postflight/$AGEMEM_EXPECTED_COMMIT/
  postflight_report.json
$TRINITY_CHECKPOINT_ROOT_DIR/m8b_logs/$AGEMEM_EXPECTED_COMMIT/
  *.log
```

## 立即停止条件

- strict runtime gate 仍有任意 SKIP；
- commit、配置 digest、fullwiki fingerprint 或固定 ID 不一致；
- 模型、数据或 checkpoint 不在 `/data/hjx/Age_mem`；
- E0/E1 smoke job 目录已含旧文件，可能触发 Trinity 隐式改名或加载陈旧状态；
- `CUDA_VISIBLE_DEVICES` 未能唯一选择恰好两张 GPU、数值选择器未配合
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`、任一选中卡总显存少于 48,000 MiB、空闲显存
  少于 47,000 MiB，或 PyTorch/NVIDIA UUID 不一致；
- E1 reward breakdown 出现 terminal/total 之外的训练奖励；
- K=2 组内 policy version 改变；
- action ID、token span、old logprobs 或 ToolTrace join 失败；
- rollout MemoryStore 串状态；
- loss、KL、reward 缺失或出现 NaN/Inf，actor update sentinel 不为 1；
- provider profile 不一致、调用未记录，或凭据出现在文件/日志；
- `global_step_1` 的 model/optimizer/extra-state/LoRA 任一产物缺失或为空；
- checkpoint eval 不是 model version 1、与训练进程执行 ID 相同，或 held-out
  taskset/样本数/任务分数不完整。

任何一项失败都不扩大样本、step、GPU 数或 seed，也不进入 E3/E4/E5。
