# 跨 chunk 与具体 ADD 示例消融（first_action_probe.v3）

前轮 v2 新提示4/4 ADD，但全部照写当前公开ADD格式例中的首句。本轮只检查首动作是否依赖该具体示例，不能替代完整顺序阅读或任务评测。

从同一冻结train history按公开长度确定选择首/中/末chunk，索引 `[0, len(chunks)//2, len(chunks)-1]`，不读取问题/gold/私有index、不按奖励挑chunk。每chunk两对seed7/8、两个条件，共12次调用、6份样本/条件。所有样本独立空记忆，不看其他chunk或共享修改，不是12个独立任务或GRPO组。

- 对照single_action_probe_v2：新单动作system+当前公开chunk的具体ADD例。
- single_action_no_example_v3：完全同一system、公开正文/ID、handles、模型、thinking、采样和C/B；仅移除具体ADD格式例。保留正文全部事实，既不遮挡第一句，也不要求ADD或强制选择非首句。

通过 `prepare --comparison single_vs_no_example_v3` 冻结v3 plan。CLI run复用单GPU后端和严格parser，非法/截断不修补。旧v1/v2默认比较和历史结果保留；重放旧plan使用对应历史提交。主环境新增boolean include_add_format_example默认True，主协议行为不变，无旧YAML/checkpoint/reward身份覆盖。

预算：单物理GPU1/TP1/BF16/utilization0.55，temperature0.6/top_p1/top_k-1、thinkingFalse、max_tokens512；12次输出总上限6144tokens，reader/optimizer/API0。CPU prepare/验证无需GPU；GPU前必须用户确认。

准备命令（新目录，不覆盖旧结果）：

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
export PROBE_DIR="$SMOKE_DIR/cross-chunk-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES= python scripts/agemem_dynamic_v2_first_action_probe.py prepare \
 --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
 --launcher "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry4.yaml" \
 --taskset "$SMOKE_DIR/taskset" --gpu-id 1 \
 --comparison single_vs_no_example_v3 --output-dir "$PROBE_DIR"
```

确认后GPU run使用 `$PROBE_DIR/plan.public.json` 和新 `$PROBE_DIR/results`，先检查卡/已有结果；不重复采样直到成功、不提高预算。

报告每条件和每chunk成功写入、格式、max-token hit、exact-body、first_sentence_count、nonfirst_fact_count。非首句精确匹配只是诊断指标，不奖励/强制它，也不意味着自由改写错误。去例仍选首句可能是位置偏好，不能直接判错误；去例保存正文与source/time可信才支持“不完全依赖具体例”的有限结论。人工审核内容/source/time，关注背景事实与动态更新；无独立Oracle注入。未验证顺序工具交互、记忆选择质量、完整流式保留、reader/END/LIFE/learning效果。
