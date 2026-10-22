from __future__ import annotations

import unittest

from teng_agent.tools.base import ToolCall, ToolRegistry
from teng_agent.tools.shop import ShopService, register_shop_tools


def make_registry():
    reg = ToolRegistry()
    register_shop_tools(reg, ShopService())
    return reg


class TestShopTools(unittest.TestCase):
    def setUp(self):
        self.reg = make_registry()

    def test_registry_has_all_tools(self):
        expected = {"query_order", "query_logistics", "apply_refund", "search_product", "search_faq"}
        self.assertTrue(expected.issubset(set(self.reg.names())))

    def test_query_order_list(self):
        r = self.reg.execute(ToolCall(tool="query_order", arguments={"user_id": "u1001"}))
        self.assertTrue(r.success)
        self.assertIn("订单列表", r.output)
        self.assertGreaterEqual(r.output["订单数"], 3)

    def test_query_order_detail(self):
        r = self.reg.execute(
            ToolCall(tool="query_order", arguments={"user_id": "u1001", "order_id": "TM20261001001"})
        )
        self.assertTrue(r.success)
        self.assertEqual(r.output["订单号"], "TM20261001001")
        self.assertIn("物流", r.output)

    def test_query_order_wrong_user(self):
        r = self.reg.execute(
            ToolCall(tool="query_order", arguments={"user_id": "u9999", "order_id": "TM20261001001"})
        )
        self.assertIn("error", r.output)

    def test_query_logistics(self):
        r = self.reg.execute(ToolCall(tool="query_logistics", arguments={"order_id": "TM20261001001"}))
        self.assertTrue(r.success)
        self.assertEqual(r.output["快递公司"], "顺丰速运")

    def test_query_logistics_no_shipment(self):
        r = self.reg.execute(ToolCall(tool="query_logistics", arguments={"order_id": "TM20261010003"}))
        self.assertIn("error", r.output)

    def test_apply_refund_success(self):
        r = self.reg.execute(
            ToolCall(tool="apply_refund", arguments={"order_id": "TM20260920002", "reason": "不想要了"})
        )
        self.assertTrue(r.success)
        self.assertIn("退款申请已提交", r.output["message"])

    def test_apply_refund_already_refunding(self):
        self.reg.execute(ToolCall(tool="apply_refund", arguments={"order_id": "TM20260920002", "reason": "x"}))
        r = self.reg.execute(
            ToolCall(tool="apply_refund", arguments={"order_id": "TM20260920002", "reason": "x"})
        )
        self.assertIn("已在退款流程", r.output["message"])

    def test_search_product(self):
        r = self.reg.execute(ToolCall(tool="search_product", arguments={"keyword": "耳机"}))
        self.assertTrue(r.success)
        self.assertGreaterEqual(r.output["找到商品"], 2)

    def test_search_product_none(self):
        r = self.reg.execute(ToolCall(tool="search_product", arguments={"keyword": "洗衣机"}))
        self.assertIn("未找到", r.output["message"])

    def test_search_faq(self):
        r = self.reg.execute(ToolCall(tool="search_faq", arguments={"query": "运费多少"}))
        self.assertTrue(r.success)
        self.assertTrue(any("运费" in h["答案"] or "包邮" in h["答案"] for h in r.output["匹配结果"]))

    def test_unknown_tool(self):
        r = self.reg.execute(ToolCall(tool="nope", arguments={}))
        self.assertFalse(r.success)


if __name__ == "__main__":
    unittest.main()
