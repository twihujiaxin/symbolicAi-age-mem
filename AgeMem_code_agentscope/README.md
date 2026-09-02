# AgeMem (AgentScope)

Standalone release of the **AgeMem** agent: a ReAct-style agent with **6 tools** for self-managing **short-term context** and **long-term memory**, built on [AgentScope](https://github.com/modelscope/agentscope).

## Features

- **6 tools**: `summary_context`, `clear_context`(`filter_context`), `retrieve_memory`, `add_memory`, `update_memory`, `delete_memory`
- **Backend-independent long-term memory**: `MemoryStore` protocol with an in-memory implementation and embedding-based retrieval
- **Rollout isolation**: one store per `rollout_id`, with cross-rollout restore protection
- **Auditable mutations**: versioned updates and research-mode soft deletes
- **Replayable trajectories**: optional strict JSONL recording with complete memory snapshots
- **Offline M3 environment**: 30 deterministic HotpotQA-style two-hop memory tasks
- **Offline M4 rewards**: Oracle AP grounding, a hand-authored DFA, and once-only milestone rewards
- **Offline M5 benchmark**: real local HotpotQA adaptation, fixed smoke splits, and auditable Oracle reports
- **Offline M6 extraction benchmark**: strict Triple/AP provenance, explicit versioned state, and Oracle error-propagation analysis

### The 6 Memory Tools

| Tool | Type | Description |
|------|------|-------------|
| `summary_context` | Short-term | Compress selected conversation rounds into a summary |
| `clear_context` / `filter_context` | Short-term | Remove irrelevant messages by similarity |
| `retrieve_memory` | Short-term | Pull relevant entries from long-term memory into context |
| `add_memory` | Long-term | Store new information in the external vector store |
| `update_memory` | Long-term | Create a new active version and supersede the old version |
| `delete_memory` | Long-term | Soft-delete an entry while retaining auditable history |

### MemoryStore and rollout isolation

`AgentScopeLongtermMemory` is an adapter between the asynchronous AgentScope
memory API and the synchronous `MemoryStore` protocol. The memory tools call the
adapter and do not import or inspect a concrete backend. The default backend is
`InMemoryStore`.

Each store is bound to exactly one `rollout_id`. Use
`RolloutMemoryStoreRegistry.get_or_create(rollout_id)` when several rollouts are
collected concurrently. A snapshot from one rollout cannot be restored into a
different rollout.

Updates append a new version and mark the previous active version as
`superseded`. In research mode, deletes append a `discarded` tombstone. Normal
retrieval returns only active versions, while `snapshot()` and
`get_memory_history()` retain the complete audit trail.

### HotpotQA-style M3 toy environment

The M3 fixture contains 30 artificial two-hop tasks split into 20 train, 5 dev,
and 5 test examples. It does not download HotpotQA and does not call an LLM or
online embedding service. It follows the existing AgeMem stage protocol:

```text
Stage 1: observe facts and build/update LTM
Stage 2: clear Stage 1 STM and inject deterministic distractors
Stage 3: retain Stage 2 context, retrieve supporting facts, and answer
```

The public `StageInput` excludes answers, supporting-fact IDs, and Oracle
labels. Gold and explicit error policies are offline test fixtures. Semantic
Oracle labels are saved in `ToolResultSnapshot.metadata["oracle_labels"]` for a
future M4 mapper; M3 does not define APs, a DFA, or logic rewards.

Generate a complete replayable gold trajectory without any model call:

```python
import asyncio

from AgeMem_code_agentscope import (
    GoldMemoryPolicy,
    ToyEpisodeRunner,
    ToyTaskDataset,
    TrajectoryRecorder,
)


async def run():
    task = ToyTaskDataset.from_json().get("toy-train-001")
    result = await ToyEpisodeRunner().run(
        task,
        GoldMemoryPolicy(),
        rollout_id="example-rollout",
        seed=7,
        recorder=TrajectoryRecorder("runs/trajectories/toy-m3.jsonl"),
    )
    assert result.episode.success


asyncio.run(run())
```

The task fixture is stored at `data/toy/hotpotqa_memory_tasks.json`. M2
`MemoryStoreSnapshot` is used for episode checkpoint/restore, and a shared
`ToyEnvironmentPool` provides one isolated store per rollout.

### M4 Oracle AP and offline DFA reward

M4 consumes the strict `oracle_labels` already stored in each M3 tool result.
It does not infer propositions from tool names: a raw or failed `Add_memory` or
`Retrieve_memory` call produces no progress AP and therefore no milestone
reward. The hand-authored positive DFA follows this sequence:

```text
q0 --stored_supporting_fact--------> q1
q1 --supporting_coverage_complete--> q2
q2 --retrieved_supporting_fact-----> q3
q3 --answered_correctly------------> q4 (accept)
```

`updated_stale_fact` is a parallel progress edge. Each progress edge is
rewarded at most once per rollout. Irrelevant stores/retrievals are recorded as
violations, and deleting a supporting fact enters the rejecting state. An
unfinished trace that reaches the configured step bound enters the timeout
state. This is a finite-trace, hand-authored DFA; M4 does not add a Critic,
natural-language extraction, LTL compilation, or a negative automaton.

Replay an M3 JSONL trajectory without a model call:

```python
from AgeMem_code_agentscope import OfflineRewardReplay, ToyTaskDataset

task = ToyTaskDataset.from_json().get("toy-train-001")
result = OfflineRewardReplay.from_config("terminal_dfa").replay_jsonl(
    "runs/trajectories/toy-m3.jsonl",
    task=task,
    rollout_id="example-rollout",
    output_path="runs/rewards/toy-m4.jsonl",
)
assert result.accepted
```

Reward coefficients and the 12-step timeout are externalized in
`configs/m4_reward.json`. The `terminal_only` and `terminal_dfa` profiles both
save environment, milestone, violation, trend, and format components. Trend
shaping is fixed to zero in M4; violations are audited with zero penalty until
a later experiment explicitly enables a non-positive weight.

### M5 real HotpotQA Oracle benchmark

M5 reads a local Hugging Face `DatasetDict` saved with `save_to_disk`; it never
downloads data at runtime. By default the adapter expects
`../data/hotpot_qa/fullwiki` relative to the repository, or the path in
`HOTPOTQA_PATH`. Install the additional reader dependency with:

```powershell
.\.venv\python.exe -m pip install -r AgeMem_code_agentscope\requirements-hotpotqa.txt
```

Run the fixed 6-train / 2-dev / 2-test smoke benchmark:

```powershell
.\.venv\python.exe -m AgeMem_code_agentscope.hotpotqa_benchmark `
  --data-path D:\Project\Age-Mem\data\hotpot_qa\fullwiki
```

Add `--rebuild-manifest` only when intentionally regenerating the deterministic
selection manifest. Smoke train comes from source train; smoke dev and test are
disjoint labeled subsets of source validation. The official source test split
is checked end-to-end for absent answers and supporting-fact labels, but is not
used for Oracle scoring.

The adapter resolves every supporting fact by the exact `(title, sent_id)`
annotation and derives a stable content-bound fact ID. It does not use title or
sentence substring matching. The public three-stage `StageInput` contains no
answer field, supporting IDs, or Oracle labels.

The benchmark records and replays M1 trajectories, then runs the M4
`terminal_dfa` profile entirely offline. `gold` is an Oracle upper bound;
`wrong_answer` and `missing_support` are deterministic failure controls. They do
not represent real base-model performance, because M5 deliberately makes zero
LLM calls. Natural-language AP extraction and model training remain out of
scope.

Outputs:

- Full context-bearing trajectory and reward JSONL: `runs/m5_hotpotqa_smoke/`
  (gitignored; treat as sensitive data).
- Reproducible split manifest: `data/splits/hotpotqa_smoke_manifest.json`.
- Compact metrics and failure audit: `artifacts/m5_hotpotqa_smoke/`.
- Human-readable report: `docs/m5_hotpotqa_oracle_benchmark.md`.

`Retrieval recall@k` is an Oracle-directed cumulative diagnostic: it measures
supporting-fact recall over the union of repeated top-1 retrieval results, where
`k` is the number of unique returned facts. It is not standard single-query
model Recall@k. Context tokens are a tokenizer-independent cumulative estimate
over every processed observation, including repeated context across timesteps.
Memory precision is computed over final active memory records. Failure audits
retain exact source pointers, memory version history, and every grounded AP/DFA
transition without copying full context or answer text.

### M6 extracted Triple/AP and explicit semantic state

M6 first audits the canonical M5 artifacts, then derives namespaced v2 action
and credit records without modifying the source JSONL. The canonical migration
covers 30 rollouts and 224 actions with an exact 224/224
`ActionEvent`-to-`ActionCreditRecord` join. Existing tool-call IDs remain the
`action_id`; the 20 actions that fire two DFA edges retain both in ordered
`transition_ids`. M5 rule, Oracle, and error-injector traces have no recoverable
token spans, token IDs, token logprobs, or policy version, so these fields stay
`None` rather than being synthesized.

Run or reproduce the non-destructive migration:

```powershell
@'
from AgeMem_code_agentscope.action_schema import migrate_m5_canonical_report

result = migrate_m5_canonical_report(
    "artifacts/m5_hotpotqa_smoke/oracle_benchmark.json",
    runtime_root="runs/m5_hotpotqa_smoke",
    output_root="runs/m6_schema_v2",
)
print(result.manifest.digest)
'@ | .\.venv\python.exe -
```

The audited manifest digest is
`3615ce1041b47ea30513e81f5ef812da4060df9fb854b843162c171443ac5452`.
The schema audit is documented in `docs/schema_audit_m6.md` and its dedicated
commit is `d1d45ab feat(agemem): audit and migrate M5 action schema`.

The extraction layer validates exact evidence spans and source digests,
finite confidence, known subjects, and a versioned category registry. Invalid
or unknown candidates are quarantined. The deterministic mock extractor and
the injected-client LLM adapter share the same strict contract; the LLM adapter
is tested with a fake client only. The reported M6 benchmark made zero real LLM
calls.

The group cache stores only action-independent candidates. Its key includes
task, split, group, stage, observation and constraint digests, plus extractor,
model, and prompt versions. Candidates are revalidated and rebound to the
original `action_id` during materialization. Relevance labels and AP records
never enter this cache.

`StateTracker` maintains a separate semantic history for every rollout.
Single-valued categories use versioned, half-open validity intervals when a
new value overwrites an old one; multi-valued categories may retain several
active values. Unknown subjects/categories, unresolved pronouns, and
same-action single-value conflicts fail closed into quarantine. Snapshot,
restore, reset, action ordering, and rollout isolation are deterministic.

AP grounding consumes only public tool results, memory before/after deltas,
validated triples, and explicit state deltas. It does not read M3/M4
`oracle_labels` or private memory-role metadata, and a bare ADD or RETRIEVE call
does not earn a semantic AP. Every derived AP contains its source `action_id`
and, where applicable, Triple/State/Memory evidence IDs. Offline reward replay
then reuses the unchanged M4 hand-authored positive DFA and once-only milestone
semantics.

The evaluation corpus contains 10 HotpotQA smoke tasks, 34 annotated sentences,
and 37 manual triples: 24 official supporting relevant facts and 10 irrelevant
samples. The loader validates each exact source split/index/title/sentence
pointer, Hotpot ID, text SHA-256, and stable fact ID without committing the
source sentence. Its corpus digest is
`fa74d5098e8dd4040d66ca99ecd76346d4cc799a59ec3f3a4133ba9bab98edd0`.

Run the fixed benchmark against the local fullwiki data:

```powershell
@'
from AgeMem_code_agentscope.memory_extraction.benchmark import (
    run_default_m6_benchmark,
)

artifacts = run_default_m6_benchmark(
    data_path=r"D:\Project\Age-Mem\data\hotpot_qa\fullwiki",
)
print(artifacts.report.digest)
'@ | .\.venv\python.exe -
```

Both profiles cover the same 30 rollouts and 224 actions. Each records 94 cache
hits, 130 misses, 164 extractor calls, and 100% AP provenance integrity.

| Profile | Triple F1 | AP F1 | False Accept | False Reject | Action reward MAE / RMSE / bias / max | Trajectory signed / abs error | Accepted / rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| `human_backed_mock` | 1.0000000000 | 0.9760765550 (FP 0, FN 10) | 0/20 | 0/10 | 0 / 0 / 0 / 0 | 0 / 0 | 10 / 20 |
| `controlled_error` | 0.8695652174 (TP 30, FP 2, FN 7) | 0.8369565217 (TP 154, FP 0, FN 60) | 0/20 | 5/10 = 0.5 | 0.056919642857 / 0.140748953307 / -0.032366071429 / 0.5 | -7.25 / 7.25 | 5 / 25 |

The human-backed mock is a Triple-extraction upper bound, not a real model or
an upper bound on the complete AP pipeline. Its ten AP false negatives are
fail-closed answer-correctness decisions; they cause no action-level or
trajectory-level reward error. The controlled profile uses synthetic
drop/corrupt errors and is not an empirical LLM error distribution. Relevance
and required coverage are supplied by a separate human Oracle semantic target
for evaluation, so this is not an end-to-end label-free AP system.

Outputs:

- Compact JSON report and action-linked error audit:
  `artifacts/m6_extraction_benchmark/`.
- Human-readable report: `docs/m6_extraction_benchmark.md`.
- Derived v2 migration plus extracted `ActionCreditRecord` and AP-record JSONL:
  `runs/m6_schema_v2/` and `runs/m6_extraction_benchmark/` (gitignored).

The final report digest is
`e803f7752dc9e7357284887cf7716273bbd5396f62db1fc438d7cad95a2f9f92`.
Run the 43 M6 tests with:

```powershell
.\.venv\python.exe -m unittest `
  tests.common.m6_schema_migration_test `
  tests.common.m6_extractor_test `
  tests.common.m6_state_tracker_test `
  tests.common.m6_grounding_reward_test `
  tests.common.m6_extraction_metrics_test `
  tests.common.m6_extraction_benchmark_test -v
```

M6 does not implement a Group Critic, GRPO, or model training.

The M6 closeout additionally audits each of the five controlled-error False
Rejects. Schema-v2 lineage, action coordinates, StateFact identity, AP
grounding, and both DFA/reward streams are checked from existing artifacts; 74
DFA action checks establish that every failure is the expected consequence of
one relevant-fact drop. Run the four closeout tests with
`tests.common.m6_false_reject_audit_test`. The audit digest is
`59a582d31396b548c0aa2c9dfc78cb5c93f6d6347a8e073d1ce0d5f291648032`.

### M7 Group Critic and offline automaton validation

M7 keeps the M4 hand-authored DFA as the primary baseline and adds strict,
evidence-grounded Group Critic schemas, a deterministic mock critic, an
injected-client LLM adapter, fail-closed validation, deterministic milestone
DAG compilation, explicit hand-DFA/terminal-only fallback, and AP-credit-only
offline replay. Replay consumes the existing `TrajectoryStepV2` and
`ActionCreditRecord`; it does not rerun extraction, StateTracker, grounding, or
an LLM.

Run the fixed offline benchmark against the local M5/M6 artifacts:

```powershell
@'
from AgeMem_code_agentscope.group_critic import write_m7_offline_report

report = write_m7_offline_report()
print(report.digest)
'@ | .\.venv\python.exe -
```

The report covers 10 real HotpotQA smoke tasks, K=3 rollouts, three AP profiles,
and 224 actions. All 90 hand-DFA profile/rollout replays exactly match their M6
action rewards. Oracle and human-backed profiles have FA/FR `0/20` and `0/10`;
the controlled-error profile retains the five explained FRs (`0/20`, `5/10`).
The Critic plus explicit-fallback pipeline and hand DFA agree on all 90
profile/rollout terminal decisions and all 3 x 224 action-reward observations.
Twenty-five cyclic outputs are rejected
by the validator and five mock outputs are unavailable; all 30 cases use an
explicit fallback, with zero silent adoption. Evidence coverage is 451/451;
150 direct repeat and 180 K-order permutation checks are stable. The hand-DFA
farming audit passes 10 duplicate-ADD and 10 two-step RETRIEVE-loop scenarios;
it does not claim Critic-DFA farming coverage.

Each critic group is bound to exactly one complete current-task HotpotQA row,
including the answer, full context, and official supporting-fact pointers. This
record is marked `critic_only_privileged/current_task_only`, participates in the
prompt and cache digest, and is not exposed through the policy observation.

Outputs are under `artifacts/m7_group_critic/` and the readable report is
`docs/m7_group_critic_offline_validation.md`. The report digest is
`87e41a0ccee6dc0dc24dfa18c898853b6530a42efdde5badbd3d0013accbac75`.
No real LLM, GRPO, or training is used.

## Install

From the folder containing `AgeMem_code_agentscope` (e.g. project root):

```bash
pip install -r AgeMem_code_agentscope/requirements.txt
```

## Run

From the **parent directory** of `AgeMem_code_agentscope` (so that `AgeMem_code_agentscope` is a package):

```bash
python -m AgeMem_code_agentscope.main
```

Example (DashScope):

```bash
export DASHSCOPE_API_KEY=your_key
python -m AgeMem_code_agentscope.main
```

## Configuration (environment variables)

| Variable | Description |
|----------|-------------|
| `AGEMEM_MODEL_PROVIDER` | Main-model backend: `dashscope` (default) or `ollama` |
| `AGENT_MODEL_NAME` | Model name, e.g. `qwen-max` or local `qwen3:4b` |
| `DASHSCOPE_API_KEY` | Api key |
| `OLLAMA_HOST` | Optional Ollama server URL; defaults to `http://localhost:11434` |
| `AGEMEM_SHOW_TOOL_TRACE` | Set to `1` to print each tool call and result in the terminal |
| `AGEMEM_TRAJECTORY_PATH` | Optional JSONL path; enables complete replayable trajectory recording |
| `AGEMEM_TASK_ID` | Task identifier written to trajectory records; defaults to `standalone-demo` |
| `AGEMEM_ROLLOUT_ID` | Optional rollout identifier; a UUID is generated when omitted |
| `HOTPOTQA_PATH` | Optional local `save_to_disk` HotpotQA fullwiki DatasetDict path used by M5/M6 |

### Show tool calls while the agent is answering

PowerShell:

```powershell
$env:AGEMEM_SHOW_TOOL_TRACE = "1"
python -m AgeMem_code_agentscope.main
```

The terminal will print the tool name, input arguments, and returned
`tool_result` as JSON. The internal `generate_response` finish tool is hidden
to avoid printing the final answer twice.

### Run Qwen3 4B locally with Ollama

The public DashScope endpoint may not expose a serverless 4B model. On a
Windows machine with limited VRAM, use Ollama's quantized `qwen3:4b`:

```powershell
ollama pull qwen3:4b
$env:AGEMEM_MODEL_PROVIDER = "ollama"
$env:AGENT_MODEL_NAME = "qwen3:4b"
$env:DASHSCOPE_API_KEY = "your-key"
python -m AgeMem_code_agentscope.main
```

The main agent runs locally. The current demo still uses DashScope for
embedding, summarization, and similarity scoring.

### Record and replay a trajectory

Trajectory recording is opt-in because the JSONL contains raw observations,
tool arguments/results, and complete memory snapshots (including embeddings).
Treat the file as sensitive data. It is intended for deterministic replay and
is not redacted or truncated.

PowerShell:

```powershell
$env:AGEMEM_TRAJECTORY_PATH = "runs/trajectories/demo.jsonl"
$env:AGEMEM_TASK_ID = "demo-task"
$env:AGEMEM_SHOW_TOOL_TRACE = "1"
python -m AgeMem_code_agentscope.main
```

The CLI prints the generated rollout ID. Query one recorded step:

```powershell
python -m AgeMem_code_agentscope.replay runs/trajectories/demo.jsonl `
  --task-id demo-task --rollout-id <printed-rollout-id> --timestep 0
```

Replay the complete memory-state sequence without calling a model or embedding
service:

```powershell
python -m AgeMem_code_agentscope.replay runs/trajectories/demo.jsonl `
  --task-id demo-task --rollout-id <printed-rollout-id> `
  --replay --require-complete
```

Each tool action is one timestep. A record contains the preceding observation,
canonical action, ToolResponse chunks, memory before/after, environment reward,
and completion flag. Memory tools use `env_reward=0.0`; future environment tools
can return `metadata["env_reward"]` in their ToolResponse.

## Layout

```
AgeMem_code_agentscope/
  __init__.py    # Package exports (AgeMem, memory, prompts)
  main.py        # Entry point (CLI), model building (DashScope / OpenAI only)
  agent.py       # AgeMem (ReAct agent + 6 tools)
  memory.py      # AgentScope adapter and embedding integration
  memory_store.py # MemoryStore protocol, versioned InMemoryStore, rollout registry
  prompts.py     # SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
  trajectory.py  # Strict TrajectoryStep, JSONL recorder, query and replay
  replay.py      # Offline trajectory query/replay CLI
  toy_hotpotqa/  # M3 task models, environment, policies and JSONL runner
  memory_oracle/ # M4 Oracle AP grounding, DFA runner and reward replay
  hotpotqa_benchmark/ # M5 local-data adapter, smoke manifest, metrics and CLI
  action_schema/ # M6 action-level v2 contracts and non-destructive migration
  memory_extraction/ # M6 extractors, cache, state, AP grounding, reward and benchmark
  group_critic/ # M7 structured Critic, validator/compiler, replay and benchmark
  src/           # Helpers: utils, llm_client, schemas, hooks
  requirements.txt
  README.md
```
