from __future__ import annotations

import unittest

from teng_agent.agent import MallAgent, build_agent
from teng_agent.llm.mock_llm import MockLLM
from teng_agent.memory import ConversationMemory
from teng_agent.tools.base import ToolRegistry
from teng_agent.tools.shop import ShopService, register_shop_tools


def make_agent() -> MallAgent:
    reg = ToolRegistry()
    register_shop_tools(reg, ShopService())
    return MallAgent(llm=MockLLM(), registry=reg, memory=ConversationMemory(max_messages=30))


class TestAgent(unittest.TestCase):
    def test_greeting(self):
        agent = make_agent()
        reply = agent.chat("你好")
        self.assertIn("您好", reply.answer)
        self.assertEqual(reply.tools_used, [])

    def test_list_orders(self):
        agent = make_agent()
        reply = agent.chat("我的订单有哪些")
        self.assertIn("TM20261001001", reply.answer)
        self.assertIn("query_order", reply.tools_used)

    def test_logistics_with_order_id(self):
        agent = make_agent()
        reply = agent.chat("查一下 TM20261001001 的物流")
        self.assertIn("顺丰", reply.answer)
        self.assertIn("query_logistics", reply.tools_used)

    def test_logistics_without_order_id(self):
        agent = make_agent()
        reply = agent.chat("我的快递到哪了")
        # Should fall back to listing orders first.
        self.assertIn("query_order", reply.tools_used)
        self.assertIn("订单", reply.answer)

    def test_refund_flow(self):
        agent = make_agent()
        reply = agent.chat("帮我退款 TM20260920002，耳机有问题")
        self.assertIn("退款", reply.answer)
        self.assertIn("apply_refund", reply.tools_used)

    def test_search_product(self):
        agent = make_agent()
        reply = agent.chat("有没有耳机推荐")
        self.assertIn("耳机", reply.answer)
        self.assertIn("search_product", reply.tools_used)

    def test_faq(self):
        agent = make_agent()
        reply = agent.chat("运费多少")
        self.assertTrue("包邮" in reply.answer or "运费" in reply.answer)
        self.assertIn("search_faq", reply.tools_used)

    def test_multi_turn_context(self):
        agent = make_agent()
        agent.chat("我的订单有哪些")
        reply = agent.chat("查一下 TM20261001001 的物流")
        self.assertIn("顺丰", reply.answer)

    def test_unknown_order(self):
        agent = make_agent()
        reply = agent.chat("查一下 TM99999999999 的物流")
        self.assertIn("未找到", reply.answer)

    def test_reset(self):
        agent = make_agent()
        agent.chat("我的订单有哪些")
        agent.reset()
        self.assertEqual(agent.memory.messages, [])

    def test_build_agent_factory(self):
        agent = build_agent()
        reply = agent.chat("你好")
        self.assertTrue(reply.answer)


if __name__ == "__main__":
    unittest.main()
