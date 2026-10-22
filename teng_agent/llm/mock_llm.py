from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from teng_agent.llm.base import BaseLLM, LLMResponse

USER_ID_RE = re.compile(r"u\d{3,}")
ORDER_ID_RE = re.compile(r"TM\d{8,}")

_INTENT_PATTERNS = [
    (
        "query_logistics",
        re.compile(r"物流|快递|到哪|发货到|运单|轨迹|什么时候到|派送"),
        lambda m: _default_args(m, order_id=True),
    ),
    (
        "apply_refund",
        re.compile(r"退款|退货|不想要|申请退|取消订单"),
        lambda m: _default_args(m, order_id=True, reason="用户申请退款"),
    ),
    (
        "query_order",
        re.compile(r"订单|下单|买过|我的订单|查一下.*单"),
        lambda m: _default_args(m),
    ),
    (
        "search_faq",
        re.compile(r"运费|包邮|发票|保修|支付方式|怎么.*退|发货时间"),
        lambda m: {"query": m.group(0)},
    ),
    (
        "search_product",
        re.compile(r"商品|多少钱|有没有|推荐|耳机|手表|键盘|充电|数据线|买"),
        lambda m: {"keyword": _extract_keyword(m.string)},
    ),
]

_KEYWORD_HINTS = ["耳机", "手表", "键盘", "充电宝", "数据线", "充电", "数码", "配件"]

_FALLBACK_FAQ = {
    "运费": "全场满99元包邮；不满99元收8元运费。",
    "退款": "已发货订单需先拒收或寄回商品，签收后7天内可申请退款。",
    "发票": "订单完成后可在订单详情页申请电子发票。",
    "发货": "现货商品当天16点前付款当天发货。",
    "支付": "支持微信支付、支付宝、银行卡及花呗分期。",
    "保修": "数码产品全国联保1年，配件保修3个月。",
}


class MockLLM(BaseLLM):
    """Rule-based mock so the full agent loop runs without any API key.

    It emitates function-calling behaviour: emit one tool_call per turn,
    then a final natural-language answer after seeing tool results.
    """

    def __init__(self, default_user_id: str = "u1001"):
        self.default_user_id = default_user_id
        self._pending: Dict[str, Any] = {}

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> LLMResponse:
        available = {t["function"]["name"] for t in (tools or [])}

        # Phase 2: tool results from the CURRENT turn only (after last user msg).
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        tool_msgs = [
            m
            for m in messages[last_user_idx + 1 :]
            if m.get("role") == "tool"
        ]
        if tool_msgs:
            return LLMResponse(content=self._compose_answer(messages, tool_msgs))

        # Phase 1: decide which tool to call.
        last_user = _last_content(messages, "user") or ""
        plan = self._plan(last_user, available)
        if plan is None:
            return LLMResponse(content=self._chit_chat(last_user))

        name, args = plan
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        )

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def _plan(self, text: str, available: set) -> tuple | None:
        user_id = self._extract_user_id(text)
        order_id = self._extract_order_id(text)

        for name, pattern, args_fn in _INTENT_PATTERNS:
            if name not in available:
                continue
            m = pattern.search(text)
            if not m:
                continue
            args = args_fn(m)
            if "user_id" in _params_of(name, available) or name == "query_order":
                args.setdefault("user_id", user_id)
            if name in ("query_logistics", "apply_refund"):
                if not order_id and not args.get("order_id"):
                    # No order id in text: list orders first.
                    if "query_order" in available:
                        return "query_order", {"user_id": user_id}
                    return name, args
                args["order_id"] = order_id or args.get("order_id")
            return name, args

        # Fallback: FAQ keywords, else list orders if user mentions self.
        for kw, answer_key in _FALLBACK_FAQ.items():
            if kw in text and "search_faq" in available:
                return "search_faq", {"query": text}
        if any(w in text for w in ("我", "查", "看")) and "query_order" in available:
            return "query_order", {"user_id": user_id}
        return None

    def _extract_user_id(self, text: str) -> str:
        m = USER_ID_RE.search(text)
        return m.group(0) if m else self.default_user_id

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        m = ORDER_ID_RE.search(text)
        return m.group(0) if m else None

    # ------------------------------------------------------------------ #
    # Answer composition
    # ------------------------------------------------------------------ #

    def _compose_answer(self, messages: List[Dict[str, Any]], tool_msgs: List[Dict[str, Any]]) -> str:
        import json

        last_user = _last_content(messages, "user") or ""
        parts: List[str] = []

        for tm in tool_msgs:
            try:
                data = json.loads(tm.get("content", "{}"))
            except json.JSONDecodeError:
                continue
            if not data.get("success", True):
                parts.append(str(data.get("output", "工具执行失败。")))
                continue
            out = data.get("output")
            if isinstance(out, dict) and "error" in out:
                parts.append(out["error"])
            elif isinstance(out, dict) and "message" in out and len(out) <= 3:
                parts.append(out["message"])
            elif isinstance(out, dict) and "商品列表" in out:
                lines = [f"- {p['名称']}：{p['价格']}（库存{p['库存']}，{p['分类']}）" for p in out["商品列表"]]
                parts.append("为您找到以下商品：\n" + "\n".join(lines))
            elif isinstance(out, dict) and "匹配结果" in out:
                best = out["匹配结果"][0]
                parts.append(f"关于「{best['问题']}」：{best['答案']}")
            elif isinstance(out, dict) and "订单列表" in out:
                lines = [
                    f"- {o['订单号']}｜{o['状态']}｜{o['商品']}｜{o['实付']}"
                    for o in out["订单列表"]
                ]
                parts.append("您目前的订单：\n" + "\n".join(lines) + "\n\n需要查看物流或退款请告诉我订单号~")
            elif isinstance(out, dict) and "订单号" in out:
                seg = [f"订单 {out['订单号']}（{out.get('状态', out.get('物流状态', ''))}）"]
                if "商品" in out:
                    seg.append(f"商品：{out['商品']}")
                if "实付" in out:
                    seg.append(f"实付：{out['实付']}")
                if "物流状态" in out:
                    seg.append(
                        f"物流：{out.get('快递公司', '')} {out.get('运单号', '')}，{out['物流状态']}，"
                        f"当前在{out.get('当前位置', '—')}，预计 {out.get('预计送达', '—')} 送达"
                    )
                if "退款金额" in out:
                    seg.append(f"退款金额：{out['退款金额']}，{out.get('预计到账', '')}")
                if "message" in out:
                    seg.append(out["message"])
                parts.append("\n".join(seg))
            elif isinstance(out, dict):
                parts.append(json.dumps(out, ensure_ascii=False))
            else:
                parts.append(str(out))

        if not parts:
            return self._chit_chat(last_user)
        return "\n\n".join(parts)

    @staticmethod
    def _chit_chat(text: str) -> str:
        if any(w in text for w in ("你好", "您好", "hi", "hello", "在吗")):
            return "您好，我是商城智能客服小腾～可以帮您查订单、查物流、申请退款、找商品或解答商城政策问题，请问有什么可以帮您？"
        if any(w in text for w in ("谢谢", "感谢", "好的")):
            return "不客气，祝您购物愉快！还有其他问题随时找我～"
        return "抱歉，我没有理解您的问题。您可以试试：\n· 查我的订单\n· 查物流 TM20261001001\n· 申请退款 TM20260920002\n· 搜索耳机\n· 运费多少"


def _last_content(messages: List[Dict[str, Any]], role: str) -> str | None:
    for m in reversed(messages):
        if m.get("role") == role and m.get("content"):
            return m["content"]
    return None


def _extract_keyword(text: str) -> str:
    for kw in _KEYWORD_HINTS:
        if kw in text:
            return kw
    return text.strip()[:6] or "商品"


def _default_args(m: re.Match, order_id: bool = False, **extra: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = dict(extra)
    if order_id:
        found = ORDER_ID_RE.search(m.string)
        if found:
            args["order_id"] = found.group(0)
    return args


def _params_of(name: str, available: set) -> set:
    return {"user_id"} if name == "query_order" else set()
