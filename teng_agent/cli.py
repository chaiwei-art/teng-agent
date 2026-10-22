from __future__ import annotations

import argparse
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from teng_agent.agent import build_agent


def _make_llm(args: argparse.Namespace):
    if args.mock:
        return None  # build_agent default = MockLLM
    if os.environ.get("OPENAI_API_KEY") or args.api_base:
        try:
            from teng_agent.llm.openai_llm import OpenAILLM

            return OpenAILLM(model=args.model, base_url=args.api_base)
        except ImportError:
            print("[warn] openai package not installed, falling back to MockLLM. pip install openai", file=sys.stderr)
            return None
    print("[info] OPENAI_API_KEY not set, using rule-based MockLLM (no API needed).", file=sys.stderr)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teng-agent", description="teng-mall 智能客服 Agent")
    parser.add_argument("--mock", action="store_true", help="强制使用规则 MockLLM（无需 API）")
    parser.add_argument("--model", default="gpt-4o-mini", help="模型名（OpenAI 兼容）")
    parser.add_argument("--api-base", default=None, help="OpenAI 兼容 API base URL")
    parser.add_argument("--db", default=None, help="自定义 shop_db.json 路径")
    parser.add_argument("--debug", action="store_true", help="显示工具调用轨迹")
    args = parser.parse_args(argv)

    agent = build_agent(llm=_make_llm(args), db_path=args.db)

    print("=== teng-mall 智能客服「小腾」===")
    print("输入问题开始对话，Ctrl+C 或输入 quit 退出。\n")
    print("示例：")
    print("  我的订单有哪些")
    print("  查一下 TM20261001001 的物流")
    print("  帮我退款 TM20260920002，质量有问题")
    print("  有没有耳机推荐")
    print("  运费多少\n")

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            return 0
        if user_input in ("/reset", "重置"):
            agent.reset()
            print("（会话已重置）\n")
            continue

        reply = agent.chat(user_input, verbose=args.debug)
        print(f"\n小腾> {reply.answer}")
        if args.debug and reply.trace:
            print(f"\n--- trace ({reply.latency_ms}ms, tools={reply.tools_used}) ---")
            print(reply.trace)
            print("---\n")
        else:
            print()


if __name__ == "__main__":
    raise SystemExit(main())
