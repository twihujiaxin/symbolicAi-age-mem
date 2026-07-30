#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgeMem (AgentScope): CLI to run the memory agent.

"""
import asyncio
import inspect
import os
from contextlib import suppress

from agentscope.formatter import DashScopeChatFormatter, OllamaChatFormatter
from agentscope.message import Msg

try:
    from agentscope.model import DashScopeChatModel, OllamaChatModel
except Exception:
    DashScopeChatModel = OllamaChatModel = None

from .agent import AgeMem


async def close_runtime_resources(model, agent) -> None:
    """在 asyncio 事件循环关闭前释放模型与辅助客户端的连接池。"""
    # OllamaChatModel / OpenAIChatModel 会持有异步 HTTP 客户端；DashScopeChatModel
    # 本身不暴露 client，但 SDK 在当前事件循环上维护共享 aiohttp session。
    model_client = getattr(model, "client", None)
    if model_client is not None:
        close = getattr(model_client, "close", None)
        if close is not None:
            with suppress(Exception):
                result = close()
                if inspect.isawaitable(result):
                    await result

    # 下面两个是同步 OpenAI 客户端，分别用于摘要/相似度和 embedding。
    auxiliary_clients = [
        getattr(getattr(agent, "chat_client", None), "client", None),
        getattr(getattr(agent, "memory_manager", None), "client", None),
    ]
    for client in auxiliary_clients:
        close = getattr(client, "close", None)
        if close is not None:
            with suppress(Exception):
                result = close()
                if inspect.isawaitable(result):
                    await result

    with suppress(Exception):
        import dashscope

        await dashscope.close_shared_aio_session()

    # aiohttp 在 Windows 上关闭 SSL 连接需要让事件循环再运行片刻，
    # 否则 ProactorEventLoop 先退出会产生 "Event loop is closed" 噪声。
    await asyncio.sleep(0.25)


def build_model():
    """根据环境变量构建主智能体模型。

    - dashscope：调用百炼公共模型或用户已经部署好的模型。
    - ollama：在本机运行量化模型，适合 Windows + 单张消费级显卡。

    摘要、相似度判断和 embedding 仍使用项目原有的 DashScope 辅助客户端，
    因此两种模式目前都需要 DASHSCOPE_API_KEY。
    """
    provider = os.getenv("AGEMEM_MODEL_PROVIDER", "dashscope").strip().lower()

    if provider == "ollama":
        model_name = os.getenv("AGENT_MODEL_NAME") or "qwen3:4b"
        model = OllamaChatModel(
            model_name=model_name,
            # CLI 在完整回答后统一打印，关闭流式传输可简化错误与资源清理。
            stream=False,
            enable_thinking=False,
            host=os.getenv("OLLAMA_HOST") or None,
        )
        return model, OllamaChatFormatter(), provider, model_name

    if provider == "dashscope":
        model_name = os.getenv("AGENT_MODEL_NAME") or "qwen-max"
        model = DashScopeChatModel(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model_name=model_name,
            enable_thinking=False,
            # 当前 CLI 不逐 token 打印；非流式模式也能避免异常时遗留 aiohttp 会话。
            stream=False,
        )
        return model, DashScopeChatFormatter(), provider, model_name

    raise ValueError(
        "AGEMEM_MODEL_PROVIDER 仅支持 'dashscope' 或 'ollama'，"
        f"当前值为：{provider!r}"
    )


async def main() -> None:
    model = None
    agent = None
    try:
        model, formatter, provider, model_name = build_model()
        sys_prompt = (
            "You are an intelligent assistant that solves complex problems by managing context and long-term memory with tools. "
            "Your job is to capture and organize any information that is helpful, relevant, or useful for the user or for solving their problems—facts, preferences, intermediate results, key decisions, and follow-up needs. "
            "Use tools to: summarize context when it gets long, clear context when starting fresh, retrieve relevant memories, add new useful information to memory, update existing memories when things change, and delete memories when they are no longer needed. "
            "Be concise and helpful; proactively store and recall what matters for the user."
        )

        agent = AgeMem(
            name="AgeMem",
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            show_tool_trace=(
                os.getenv("AGEMEM_SHOW_TOOL_TRACE", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        )

        print("=== AgeMem (AgentScope) ===\n")
        print(f"Model provider: {provider}; model: {model_name}\n")
        print(
            "Tool trace: "
            f"{'on' if agent.show_tool_trace else 'off'} "
            "(set AGEMEM_SHOW_TOOL_TRACE=1 to enable)\n"
        )
        print("Type your message and press Enter. 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_input = input("[user]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Bye.")
                break
            try:
                reply_msg = await agent.reply(
                    msg=Msg(name="user", content=user_input, role="user")
                )
            except RuntimeError as exc:
                if "Model not exist" in str(exc):
                    print(
                        "\n[配置错误] 当前端点无法调用该模型名称。"
                        "\n- DashScope 公共服务请换成账户所在地域可用的模型；"
                        "\n- 本地 4B 请设置 AGEMEM_MODEL_PROVIDER=ollama "
                        "及 AGENT_MODEL_NAME=qwen3:4b；"
                        "\n- 百炼专属部署请填写控制台返回的 deployed_model ID。\n"
                    )
                    break
                raise
            print(f"[agent] {reply_msg.get_text_content()}\n")
    finally:
        await close_runtime_resources(model, agent)


if __name__ == "__main__":
    asyncio.run(main())
