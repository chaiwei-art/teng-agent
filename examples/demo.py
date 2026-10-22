"""Non-interactive demo: showcases the full agent loop with traces."""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from teng_agent.agent import build_agent


def main() -> None:
    agent = build_agent()
    scenarios = [
        "我的订单有哪些",
        "查一下 TM20261001001 的物流",
        "有没有耳机推荐",
        "运费多少",
        "帮我退款 TM20260920002，买重复了",
    ]
    for q in scenarios:
        reply = agent.chat(q)
        print(f"{'='*60}\n你> {q}")
        print(f"小腾> {reply.answer}")
        print(f"--- tools: {reply.tools_used} | {reply.latency_ms}ms ---\n")


if __name__ == "__main__":
    main()
