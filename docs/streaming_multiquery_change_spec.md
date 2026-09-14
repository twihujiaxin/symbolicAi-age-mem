# Streaming Multiquery Change Spec

本路径是 `streaming_multiquery_v1` 的仓库内规范入口。用户提供的完整、未删改规范保存在仓库根目录：

- [`../AGEMEM_STREAMING_MULTIQUERY_CHANGE_SPEC.md`](../AGEMEM_STREAMING_MULTIQUERY_CHANGE_SPEC.md)

实施代码不得从本文档动态读取配置，也不得把本文档中的 gold、示例或说明文本送入 policy observation。实际配置以 `configs/stream_mq/` 中通过 schema 校验的独立文件为准，实施状态与实测证据见：

- [`streaming_multiquery_implementation_report.md`](streaming_multiquery_implementation_report.md)
- [`../STATUS.md`](../STATUS.md)

此入口采用引用而不是复制，以避免两份 1,000 行以上的规范在后续修改时产生静默漂移。根规范仍是唯一正文；本文件不是缩写版，也不改变其任何要求。
