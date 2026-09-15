# v2 baseline fidelity

当前仅完成 CPU 受控基线定义；所有模型基线均为未运行。

| 基线 | 当前实现／适配 | 可见信息 | C/B | 当前状态 |
|---|---|---|---|---|
| Question-only | 待接固定 reader | 仅问题 | 同主臂 C；无 memory | 未运行 |
| Last-window | 待接 v1 snapshot tail | 问题＋合法 C_tail | 同主臂 C/B | 未运行 |
| FIFO-memory | 复用 v1 FIFO 与固定词法 retriever | active memory only | 同主臂 C/B | 环境契约已测；模型未运行 |
| Frozen-policy | 待锁同一 Qwen 初始化 | 与训练臂相同 | 同主臂 C/B | 未运行 |
| Rule timeline | `env-smoke` 的 `eager_timeline`；从公开 source text 构造，但脚本在 fixture 中用 private 映射定位关键事件 | 受控 fixture 的公开正文；实现使用特权定位 | debug C=512/B=1024 | 只能称 privileged fixture reference，不能作为公平模型基线 |
| Gold-support | 待接固定 reader 诊断 | per-query private support | 同 C | 未运行，非公平策略基线 |
| Full-store RAG | 未实现 | 若实现需明确放宽 B | 不同预算 | 未运行 |

外部 Mem-α、UMA、Mem-T、Memory-T1、Memory-R2 均未在本轮复现。任何后续适配必须记录原文检索权限、模型、tokenizer、训练量和预算；memory-only UMA 只能标为 adapted 版本。
