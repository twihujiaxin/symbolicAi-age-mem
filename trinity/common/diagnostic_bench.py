"""Shared opt-in boundary for bench-only Experience audit persistence."""


DIAGNOSTIC_BENCH_WORKFLOWS = frozenset({
    "AgeMem_hotpot_workflow_training",
    "AgeMem_dynamic_multiquery_v2_training",
})


def should_persist_diagnostic_bench(mode, workflow_type) -> bool:
    return mode == "bench" and workflow_type in DIAGNOSTIC_BENCH_WORKFLOWS
