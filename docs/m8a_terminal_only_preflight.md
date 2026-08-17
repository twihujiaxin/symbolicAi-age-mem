# M8a Terminal-only 上卡前门禁

更新时间：2026-08-17

## 结论

M8a 已建立 E1 terminal-only 的本地静态/离线契约，但尚未执行真实模型、GPU、优化器更新或 checkpoint 重载。因此本阶段的结论是“可以进入 AutoDL 两卡 smoke 前置验证”，不是“E1 已复现”，更不是“可以直接开始全量训练”。

本地未调用真实 LLM、embedding 服务或网络。

## 已实现

- `TaskFileReader` 可以读取 Hugging Face `save_to_disk` 的 `DatasetDict`，按 M5 manifest 中固定的 6 个 source index 保序取样，并在运行时核对 source fingerprint 与 6 个 Hotpot ID；
- `e1_terminal_only` 使用确定性的 HotpotQA answer F1，另行记录 EM、precision、recall 和 F1；reward breakdown 只有 terminal 与 total；
- E1 运行时强制 `multi_step_grpo + step_wise_grpo + K>=2`，DFA milestone 在 M8a 保持关闭；
- Stage 2 使用固定干扰文本，不调用 DistractorGenerator；
- Trinity workflow 复用 M2 的版本化、soft-delete、rollout-scoped MemoryStore；
- 在线工具动作保存稳定 `action_id`、完整响应 token IDs、逐 token old logprobs、token span、工具 trace 关联和冻结 policy version；
- `ActionEvent` 必须与 `Experience` 的 task/rollout/stage/timestep 完整对齐；规则、Oracle、random 与 error-injector 轨迹禁止进入 on-policy buffer；
- AgeMem ExperiencePipeline 在 operator 前和最终 output write 前各校验一次动作/credit 契约；operator 删除契约、篡改 character/token span 或 ToolTrace join 都会 fail closed；
- K 条 rollout 采样前后 policy version 不一致时，整组丢弃；
- 初始 LoRA 路径为 `null`，由 Trinity 在完整运行环境中从基座模型创建，不再引用不存在的默认目录。

对应 smoke 配置：`examples/agemem_hotpotqa/agemem_e1_dry_run.yaml`。

## 本地验证

```text
M8a tests：46 discovered，43 PASS，3 SKIP
M1～M7 related regression：145/145 PASS
existing tool-trace regression：28/28 PASS
Ruff check：PASS
Python compile：PASS
```

3 个跳过项是 WorkflowRunner/ExperiencePipeline 的 Trinity runtime 接线测试；当前 Windows Python 3.10 环境没有 PyTorch、Ray 和 vLLM。核心 token/span、logprob、policy-freeze、buffer admission、数据、reward、干扰和 MemoryStore 契约均已在本地执行。跳过项必须在 AutoDL 完整 Linux 环境补跑，不能按通过处理。

## 尚未关闭的正式实验门禁

E1 的 terminal reward 与固定干扰不调用辅助 LLM，但现有 memory workflow 仍有两类外部依赖：

- ADD/UPDATE/RETRIEVE 的 embedding 默认调用 DashScope；
- SUMMARY/FILTER 及 E2 judge 使用 `qwen-max` 路径。

因此不能把当前 E1 描述为“端到端无辅助 LLM/无网络”。正式 E1/E3 对照前必须二选一并冻结：

1. 保留上游 DashScope 路径：固定模型/请求参数，记录调用、错误、延迟和费用，并确保各实验臂完全一致；
2. 改成本地冻结 provider：先在独立 smoke 上验证检索与工具语义不发生不可接受的漂移。

这项选择会改变实验环境，不在 M8a 中擅自决定。

M8a 也尚未把 M6 Extracted AP/DFA reward 写回在线 ActionCreditRecord；该接线属于 E3/E4。E5 的 DFA-state action bucket、RTG、动作内 token 平均和动作间平均 loss 仍未实现。

## AutoDL 推荐顺序

1. 整理并提交当前 M6/M7/M8a 工作区；迁移前检查 ignored 文件。仓库根目录现有本地凭据文件不得上传，相关 key 应先轮换，云端只使用环境变量或 AutoDL 密钥配置。
2. 将 HotpotQA `fullwiki` 单独传到持久盘，核对目录、DatasetDict split、90,447 条 train 数量及 M5 manifest 的 6 个 Hotpot ID。
3. 首轮使用同机 `2 x 80GB`：1 张 rollout、1 张 trainer。不要直接按 8 卡正式模板启动。
4. 固定代码 commit、Python 3.10、CUDA/driver、PyTorch、Ray、vLLM、Transformers、Trinity 本地 editable 版本，并保存 `pip freeze` 与 GPU 信息。
5. 设置持久目录：

   ```bash
   export TRINITY_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
   export HOTPOTQA_PATH=/root/autodl-tmp/data/hotpot_qa/fullwiki
   export TRINITY_CHECKPOINT_ROOT_DIR=/root/autodl-tmp/checkpoints
   ```

6. 在启动 Ray 前先运行 M8a、tool-trace 和 M1～M7 回归；确认本地跳过的 3 个 runtime 测试在 AutoDL 变为 PASS。
7. 执行 Trinity config validation 和数据读取 smoke，确认 6 条固定样本、2-GPU 资源公式、LoRA 初始化目录、buffer 路径和 checkpoint 路径。
8. 依次执行：E0 冻结评测 → E1 单 batch/单次参数更新 → checkpoint 保存 → 新进程重载 → 同一冻结 split 评测。任何一步失败都不扩大数据或步数。
9. E1 重复运行稳定后，按 E3 → E4 → E5 推进；E2 只作为独立 heuristic dense reward 对照，M7 Critic/E6 暂不进入主训练。
10. 正式实验再扩大到 `4 x 80GB`、多个 seed，并保存每个实验臂的 commit、config、split digest、provider 配置、训练日志和冻结评测结果。

## 首次 GPU smoke 停止条件

出现以下任一情况立即停止，不继续租卡长跑：

- fullwiki fingerprint、固定 6 条 source index 或 Hotpot ID 不一致；
- 同一 K 组 policy version 变化；
- response token IDs、old logprobs、action span 或 action_id join 失败；
- rollout 间 MemoryStore 可见；
- terminal reward 含工具、记忆、context、timeout 或 DFA 分量；
- loss/KL/reward 出现 NaN/Inf；
- checkpoint 不能在新进程加载；
- 外部 provider 配置在实验臂之间不一致或调用未被记录。
