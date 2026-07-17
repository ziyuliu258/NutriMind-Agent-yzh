"""NutriMind 智能体终端聊天入口。"""

import asyncio
import uuid

from app.services.agent_graph import run_agent


async def chat() -> None:
    """启动一个内存多轮对话会话。"""
    session_id = f"cli-{uuid.uuid4()}"
    print("NutriMind 已启动。输入“退出”结束对话。")

    while True:
        try:
            message = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            return

        if message.lower() in {"exit", "quit", "退出"}:
            print("对话结束。")
            return
        if not message:
            continue

        result = await run_agent(
            session_id=session_id,
            user_message=message,
            user_id=1,
        )
        print(f"\nNutriMind：{result['response']}")

        if result.get("tool_calls"):
            names = ", ".join(call["name"] for call in result["tool_calls"])
            print(f"[调用工具：{names}]")


def start() -> None:
    asyncio.run(chat())


if __name__ == "__main__":
    start()
