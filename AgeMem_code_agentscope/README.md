# AgeMem (AgentScope)

Standalone release of the **AgeMem** agent: a ReAct-style agent with **6 tools** for self-managing **short-term context** and **long-term memory**, built on [AgentScope](https://github.com/modelscope/agentscope).

## Features

- **6 tools**: `summary_context`, `clear_context`(`filter_context`), `retrieve_memory`, `add_memory`, `update_memory`, `delete_memory`
- **Backend-independent long-term memory**: `MemoryStore` protocol with an in-memory implementation and embedding-based retrieval
- **Rollout isolation**: one store per `rollout_id`, with cross-rollout restore protection
- **Auditable mutations**: versioned updates and research-mode soft deletes
- **Replayable trajectories**: optional strict JSONL recording with complete memory snapshots

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
  src/           # Helpers: utils, llm_client, schemas, hooks
  requirements.txt
  README.md
```
