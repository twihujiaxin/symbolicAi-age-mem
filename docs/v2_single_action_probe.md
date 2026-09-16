# 单动作提示对照（first_action_probe.v2）

前轮 c4f7fcb 实测：旧提示 4 NEXT；task-explicit 提示 3 NEXT 和 1 多 ADD 截断，均无成功写入。截断响应中的 7 个完整 ADD 与公开正文/来源/时间一致，但没有执行，不能作为成功记忆。保留旧 plan、报告、日志。

本轮比较 `task_vs_single_v2`：task_explicit_probe_v1 对照 single_action_probe_v2。新提示只追加“本次一个动作、写入时选择一个具体事实、结束响应、工具反馈后再决定下一动作”的说明，不要求 ADD、不强制顺序。其余公开输入、格式例、模型/thinking、预算和 bench sampling 不变。两条件各4次，seed7–10匹配，八份独立空记忆；不是 GRPO 组或8个独立任务。

`prepare --comparison task_vs_single_v2` 生成独立 v2 身份。未指定 comparison 时保持旧 v1 两条件及 plan 结构；旧数据/主提示/旧 parser/reward 不变。代码身份检查仍严格：历史 v1 plan 若需重放，要使用其原始提交，不手改 digest。

CPU prepare（新输出目录，不需要 GPU）：

```bash
source /data/conda/etc/profile.d/conda.sh
conda activate /data/hjx/Age_mem/conda-envs/agemem-m8b
cd /data/hjx/Age_mem/AgeMem
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
export PROBE_DIR="$SMOKE_DIR/single-action-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES= python scripts/agemem_dynamic_v2_first_action_probe.py prepare \
  --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
  --launcher "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry4.yaml" \
  --taskset "$SMOKE_DIR/taskset" --gpu-id 1 \
  --comparison task_vs_single_v2 --output-dir "$PROBE_DIR"
```

GPU 运行前需用户确认。预算仍为单张物理 GPU1 A6000，TP1/BF16/utilization0.55，8×最多512输出tokens（上限4096），reader/optimizer/API调用均0。确认后按旧文档 run 子命令使用新的 plan/results；不得覆盖旧日志或放宽 parser、增加tokens、修补多动作/截断响应。

判断必须分别报告格式合规、截断、成功写入与正文/source/time人工核验。全部NEXT或零写入保留负结果；首动作改变不是全流式记忆成功，更不是学习有效。完整权重 SHA、跨任务稳定性、reader表现、END/LIFE仍未验证。
