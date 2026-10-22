from __future__ import annotations

import unittest

from teng_agent.memory import ConversationMemory


class TestMemory(unittest.TestCase):
    def test_add_and_context(self):
        mem = ConversationMemory(max_messages=10)
        mem.add("user", "你好")
        mem.add("assistant", "您好！")
        ctx = mem.as_context("sys")
        self.assertEqual(ctx[0]["content"], "sys")
        self.assertEqual(len(ctx), 3)
        self.assertEqual(ctx[1]["role"], "user")

    def test_trim_creates_summary(self):
        mem = ConversationMemory(max_messages=6)
        for i in range(10):
            mem.add("user", f"问题{i}")
            mem.add("assistant", f"回答{i}")
        self.assertLessEqual(len(mem.messages), 6)
        self.assertTrue(mem.summary)
        ctx = mem.as_context("sys")
        self.assertIn("对话摘要", ctx[1]["content"])

    def test_tool_result_message(self):
        mem = ConversationMemory()
        mem.add_tool_result("call_1", '{"ok": true}')
        self.assertEqual(mem.messages[0].role, "tool")
        self.assertEqual(mem.messages[0].tool_call_id, "call_1")

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "hi")
        mem.summary = "old"
        mem.clear()
        self.assertEqual(mem.messages, [])
        self.assertEqual(mem.summary, "")


if __name__ == "__main__":
    unittest.main()
