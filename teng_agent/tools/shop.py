from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from teng_agent.tools.base import ToolRegistry

DEFAULT_DB = Path(__file__).resolve().parent.parent / "mock_data" / "shop_db.json"

STATUS_ZH = {
    "pending": "待付款",
    "shipped": "已发货",
    "delivered": "已签收",
    "refunding": "退款中",
    "refunded": "已退款",
    "cancelled": "已取消",
}


def load_db(path: str | Path | None = None) -> Dict[str, Any]:
    db_path = Path(path) if path else DEFAULT_DB
    return json.loads(db_path.read_text(encoding="utf-8"))


class ShopService:
    """Domain layer over the mock shop database."""

    def __init__(self, db_path: str | Path | None = None):
        self.db = load_db(db_path)

    def orders_of(self, user_id: str) -> List[Dict[str, Any]]:
        return [o for o in self.db["orders"] if o["user_id"] == user_id]

    def find_order(self, order_id: str) -> Dict[str, Any] | None:
        for o in self.db["orders"]:
            if o["order_id"] == order_id:
                return o
        return None

    def search_products(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        kw = keyword.lower()
        hits = []
        for p in self.db["products"]:
            haystack = (p["name"] + " " + p["category"] + " " + " ".join(p["tags"])).lower()
            if kw in haystack:
                hits.append(p)
        return hits[:limit]

    def find_faq(self, query: str) -> List[Dict[str, Any]]:
        hits = []
        for item in self.db["faq"]:
            score = _overlap(query, item["q"])
            if score > 0:
                hits.append((score, item))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in hits[:2]]


def _overlap(a: str, b: str) -> int:
    sa, sb = set(a.lower()), set(b.lower())
    return len(sa & sb)


def _fmt_order(order: Dict[str, Any]) -> Dict[str, Any]:
    status = STATUS_ZH.get(order["status"], order["status"])
    items = "、".join(f"{it['name']}×{it['qty']}" for it in order["items"])
    out: Dict[str, Any] = {
        "订单号": order["order_id"],
        "状态": status,
        "商品": items,
        "实付": f"¥{order['total']}",
        "下单时间": order["created_at"],
    }
    if order.get("logistics"):
        lg = order["logistics"]
        out["物流"] = {
            "快递公司": lg["carrier"],
            "运单号": lg["tracking_no"],
            "物流状态": lg["status"],
            "当前位置": lg.get("location", ""),
            "预计送达": lg.get("eta", ""),
        }
    return out


def register_shop_tools(registry: ToolRegistry, service: ShopService) -> None:
    @registry.tool(
        name="query_order",
        description="查询用户的订单列表或指定订单详情。当用户提供订单号时传 order_id，否则用 user_id 查全部订单。",
        parameters={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户ID，如 u1001"},
                "order_id": {"type": "string", "description": "订单号，如 TM20261001001，可选"},
            },
            "required": ["user_id"],
        },
    )
    def query_order(user_id: str, order_id: str | None = None) -> Any:
        if order_id:
            order = service.find_order(order_id)
            if not order or order["user_id"] != user_id:
                return {"error": "未找到该订单，请确认订单号是否正确。"}
            return _fmt_order(order)
        orders = service.orders_of(user_id)
        if not orders:
            return {"error": "该用户暂无订单记录。"}
        return {
            "订单数": len(orders),
            "订单列表": [
                {
                    "订单号": o["order_id"],
                    "状态": STATUS_ZH.get(o["status"], o["status"]),
                    "商品": "、".join(f"{it['name']}×{it['qty']}" for it in o["items"]),
                    "实付": f"¥{o['total']}",
                }
                for o in orders
            ],
        }

    @registry.tool(
        name="query_logistics",
        description="查询指定订单的物流轨迹。需要 order_id。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
            },
            "required": ["order_id"],
        },
    )
    def query_logistics(order_id: str) -> Any:
        order = service.find_order(order_id)
        if not order:
            return {"error": "未找到该订单。"}
        if not order.get("logistics"):
            return {"error": f"订单 {order_id} 当前状态为“{STATUS_ZH.get(order['status'], order['status'])}”，暂无物流信息。"}
        lg = order["logistics"]
        return {
            "订单号": order_id,
            "快递公司": lg["carrier"],
            "运单号": lg["tracking_no"],
            "物流状态": lg["status"],
            "当前位置": lg.get("location", ""),
            "预计送达": lg.get("eta", ""),
        }

    @registry.tool(
        name="apply_refund",
        description="为已签收或已发货订单发起退款申请。需要 order_id 和 reason。",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "reason": {"type": "string", "description": "退款原因"},
            },
            "required": ["order_id", "reason"],
        },
    )
    def apply_refund(order_id: str, reason: str) -> Any:
        order = service.find_order(order_id)
        if not order:
            return {"error": "未找到该订单。"}
        if order["status"] == "refunding":
            return {"message": f"订单 {order_id} 已在退款流程中，请耐心等待。"}
        if order["status"] == "refunded":
            return {"message": f"订单 {order_id} 已完成退款。"}
        if order["status"] == "pending":
            return {"message": f"订单 {order_id} 尚未付款，可直接取消订单，无需退款。"}
        order["status"] = "refunding"
        return {
            "message": f"退款申请已提交（原因：{reason}）。",
            "订单号": order_id,
            "退款金额": f"¥{order['total']}",
            "预计到账": "审核通过后1-3个工作日原路退回",
            "提示": "如商品已签收，客服将联系您确认寄回地址。",
        }

    @registry.tool(
        name="search_product",
        description="按关键词搜索商品，返回名称、价格、库存。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "商品名称或关键词，如 耳机"},
                "limit": {"type": "integer", "description": "返回数量，默认3"},
            },
            "required": ["keyword"],
        },
    )
    def search_product(keyword: str, limit: int = 3) -> Any:
        hits = service.search_products(keyword, limit=max(limit, 1))
        if not hits:
            return {"message": f"未找到与“{keyword}”相关的商品，换个关键词试试？"}
        return {
            "找到商品": len(hits),
            "商品列表": [
                {
                    "名称": p["name"],
                    "价格": f"¥{p['price']}",
                    "库存": p["stock"],
                    "分类": p["category"],
                }
                for p in hits
            ],
        }

    @registry.tool(
        name="search_faq",
        description="在商城 FAQ 知识库中检索常见问题答案，如运费、发货、退款政策。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的问题，如 运费多少"},
            },
            "required": ["query"],
        },
    )
    def search_faq(query: str) -> Any:
        hits = service.find_faq(query)
        if not hits:
            return {"message": "知识库中暂未找到相关答案，已为您转接人工客服。"}
        return {"匹配结果": [{"问题": h["q"], "答案": h["a"]} for h in hits]}
