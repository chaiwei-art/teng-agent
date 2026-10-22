from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from teng_agent.llm.base import BaseLLM
from teng_agent.memory import ConversationMemory
from teng_agent.tools.base import ToolCall, ToolRegistry

SYSTEM_PROMPT = """你是“小腾”，teng-mall 商城的智能客服助手。

职责范围：查订单、查物流、申请退款、搜索商品、解答商城政策（运费/发货/发票/保修等）。

规则：
1. 先用工具获取真实数据，不要编造订单状态、价格或物流信息。
2. 回答用中文，语气友好简洁，像真人客服。
3. 工具返回错误时，如实告知并给出下一步建议（如核对订单号、转人工）。
4. 涉及金额、退款等敏感操作，回复中要复述关键信息让用户确认。
5. 与业务无关的问题，礼貌说明只提供商城相关服务。
"""


@dataclass
class AgentStep:
    kind: str  # thought | tool_call | tool_result | final
    content: str = ""
    tool: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class AgentReply:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    latency_ms: int = 0

    @property
    def trace(self) -> str:
        lines = []
        for s in self.steps:
            if s.kind == "tool_call":
                lines.append(f"[tool→] {s.tool}({json.dumps(s.arguments, ensure_ascii=False)})")
            elif s.kind == "tool_result":
                flag = "ok" if s.success else "err"
                lines.append(f"[{flag}←] {s.content}")
        return "\n".join(lines)


class MallAgent:
    """ReAct-style agent: LLM plans → tools execute → LLM answers."""

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_rounds: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.llm = llm
        self.registry = registry
        self.memory = memory or ConversationMemory()
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt

    def chat(self, user_input: str, verbose: bool = False) -> AgentReply:
        t0 = time.time()
        steps: List[AgentStep] = []
        tools_used: List[str] = []

        self.memory.add("user", user_input)
        # Turn-local messages: history + current user; grows with tool traffic.
        turn_messages: List[Dict[str, Any]] = self.memory.as_context(self.system_prompt)

        final_answer = ""
        for _round in range(self.max_rounds):
            response = self.llm.chat(turn_messages, tools=self.registry.schemas())

            if not response.tool_calls:
                final_answer = response.content or "抱歉，我暂时无法回答这个问题。"
                self.memory.add("assistant", final_answer)
                break

            # OpenAI protocol: assistant(tool_calls) must precede tool results.
            turn_messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
            )

            for tc in response.tool_calls:
                call = ToolCall(
                    tool=tc["function"]["name"],
                    arguments=tc.get("function", {}).get("arguments") or {},
                    call_id=tc.get("id", ""),
                )
                steps.append(
                    AgentStep(
                        kind="tool_call",
                        tool=call.tool,
                        arguments=call.arguments,
                    )
                )
                tools_used.append(call.tool)

                result = self.registry.execute(call)
                result_text = result.to_text()
                steps.append(
                    AgentStep(
                        kind="tool_result",
                        content=result_text[:400],
                        tool=call.tool,
                        success=result.success,
                    )
                )
                turn_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": result_text,
                    }
                )
        else:
            final_answer = "抱歉，处理您的请求超出了我的能力范围，已为您转接人工客服。"
            self.memory.add("assistant", final_answer)

        latency = int((time.time() - t0) * 1000)
        return AgentReply(
            answer=final_answer,
            steps=steps,
            tools_used=tools_used,
            latency_ms=latency,
        )

    def reset(self) -> None:
        self.memory.clear()


def build_agent(
    llm: BaseLLM | None = None,
    db_path: str | None = None,
) -> MallAgent:
    """Factory: default MockLLM + shop tools, or pass your own LLM."""
    from teng_agent.tools.shop import ShopService, register_shop_tools

    if llm is None:
        from teng_agent.llm.mock_llm import MockLLM

        llm = MockLLM()

    registry = ToolRegistry()
    register_shop_tools(registry, ShopService(db_path))
    return MallAgent(llm=llm, registry=registry)
