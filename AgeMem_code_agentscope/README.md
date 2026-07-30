# AgeMem (AgentScope)

Standalone release of the **AgeMem** agent: a ReAct-style agent with **6 tools** for self-managing **short-term context** and **long-term memory**, built on [AgentScope](https://github.com/modelscope/agentscope).

## Features

- **6 tools**: `summary_context`, `clear_context`(`filter_context`), `retrieve_memory`, `add_memory`, `update_memory`, `delete_memory`
- **Long-term memory**: in-memory vector store with embedding-based retrieval (DashScope text-embedding by default)

### The 6 Memory Tools

| Tool | Type | Description |
|------|------|-------------|
| `summary_context` | Short-term | Compress selected conversation rounds into a summary |
| `clear_context` / `filter_context` | Short-term | Remove irrelevant messages by similarity |
| `retrieve_memory` | Short-term | Pull relevant entries from long-term memory into context |
| `add_memory` | Long-term | Store new information in the external vector store |
| `update_memory` | Long-term | Update an existing memory entry |
| `delete_memory` | Long-term | Delete an obsolete memory entry |

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

## Layout

```
AgeMem_code_agentscope/
  __init__.py    # Package exports (AgeMem, memory, prompts)
  main.py        # Entry point (CLI), model building (DashScope / OpenAI only)
  agent.py       # AgeMem (ReAct agent + 6 tools)
  memory.py      # AgentScopeLongtermMemory, InMemoryVectorStore
  prompts.py     # SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
  src/           # Helpers: utils, llm_client, schemas, hooks
  requirements.txt
  README.md
```
