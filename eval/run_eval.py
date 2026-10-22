"""Offline eval: golden test set covering every tool + edge cases.

Run: python eval/run_eval.py
Prints per-case pass/fail and an overall score. Useful as a regression
gate when you swap MockLLM for a real LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teng_agent.agent import build_agent

CASES = [
    # (user_input, expected_substring_or_None, expected_tool)
    ("你好", "您好", None),
    ("我的订单有哪些", "TM20261001001", "query_order"),
    ("查一下 TM20261001001 的物流", "顺丰", "query_logistics"),
    ("我的快递到哪了", "订单", "query_order"),
    ("帮我退款 TM20260920002，质量有问题", "退款", "apply_refund"),
    ("有没有耳机推荐", "耳机", "search_product"),
    ("运费多少", "包邮", "search_faq"),
    ("怎么开发票", "发票", "search_faq"),
    ("查一下 TM99999999999 的物流", "未找到", "query_logistics"),
    ("谢谢", "不客气", None),
]


def run() -> int:
    agent = build_agent()
    passed = 0
    print(f"Running {len(CASES)} eval cases...\n")
    for i, (q, expect, tool) in enumerate(CASES, 1):
        agent.reset()
        reply = agent.chat(q)
        ok_sub = expect is None or expect in reply.answer
        ok_tool = tool is None or tool in reply.tools_used
        ok = ok_sub and ok_tool
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] #{i} Q: {q}")
        if not ok:
            print(f"       expected '{expect}' in answer / tool={tool}")
            print(f"       got: {reply.answer[:120]!r} tools={reply.tools_used}")
    score = passed / len(CASES)
    print(f"\nScore: {passed}/{len(CASES)} = {score:.0%}")
    return 0 if score >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(run())
