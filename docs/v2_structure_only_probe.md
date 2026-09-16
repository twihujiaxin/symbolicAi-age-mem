# v4 无具体事实的结构级说明诊断

动机：v3去例能保存事实，但存在ADD缺memory_id、实体/关系/值/时间放在额外参数、JSON后附文字/标点等问题。先检查结构提示，不直接归因于奖励设计或启动训练。

独立 comparison=`no_example_vs_structure_v4`，版本first_action_probe.v4：

- 对照：single_action_no_example_v3，原单动作system、无具体ADD例。
- 处理：structure_only_no_example_v4，仅追加必填memory_id/content/source_refs、完整事实及时间放content、单元素数组、数组外无正文/句点的说明和不含实际事实的字段形状。

字段形状采用显式尖括号占位符，要求替换，不能将其复制成ID或事实。没有指定要保存的实体/事实、来源或gold；所有实际来源都来自当前公开chunk。ADD不强制，NEXT仍合法。严格parser/环境执行不变，不修补缺字段、额外文字或截断响应。

继续使用同一冻结train history首/中/末chunk，每chunk2对seed7/8，12次独立空memory首动作。模型/Qwen3-4B/thinkingFalse/temperature0.6/C4096/B2048/max512全部不变。每条件6样本不是6个独立history或GRPO组；不构成顺序阅读/检索必要性/END/LIFE/学习验收。

CPU prepare不需要GPU，使用新目录：

```bash
export SMOKE_DIR=/data/hjx/Age_mem/runtime-v2-smoke-783d785-20260916-112222
export PROBE_DIR="$SMOKE_DIR/structure-only-probe-$(git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES= python scripts/agemem_dynamic_v2_first_action_probe.py prepare \
 --config "$SMOKE_DIR/dynamic_runtime_smoke.json" \
 --launcher "$SMOKE_DIR/dynamic_v2_k2m2_smoke_retry4.yaml" \
 --taskset "$SMOKE_DIR/taskset" --gpu-id 1 \
 --comparison no_example_vs_structure_v4 --output-dir "$PROBE_DIR"
```

GPU前用户确认：单物理GPU1/TP1/BF16/utilization0.55，12×最多512输出tokens=6144上限，无reader/optimizer/API。确认后run子命令使用新plan/results；不覆盖旧结果或扩大预算。

重点审计：placeholder是否误抄、memory_id是否必填、content是否包含完整事实/time、source是否真的支持正文、JSON格式/环境执行/截断、逐chunk选择及首句偏好。精确匹配失败不能直接判语义错误（v3有“。”→“.”）。报告实际运行和负结果，跨批次相同seed不保证已观测输出相同。

下一步有限顺序阅读只能在接口可解释且模型输出可信后单独冻结配置/预算；本轮没有运行或声称实现新的顺序rollout验收。不能用首动作写入改善推断整段记忆/reader F1/训练有效；主协议、历史profile和旧产物保持可复现。
