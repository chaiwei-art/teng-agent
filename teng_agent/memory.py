from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class ConversationMemory:
    """Sliding-window memory with optional rolling summary."""

    def __init__(self, max_messages: int = 20, max_summary_chars: int = 500):
        self.messages: List[Message] = []
        self.summary = ""
        self.max_messages = max_messages
        self.max_summary_chars = max_summary_chars

    def add(self, role: str, content: str, **kwargs: Any) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self._trim()
        return msg

    def add_tool_result(self, tool_call_id: str, content: str) -> Message:
        return self.add(role="tool", content=content, tool_call_id=tool_call_id)

    def _trim(self) -> None:
        if len(self.messages) <= self.max_messages:
            return
        overflow = self.messages[: len(self.messages) - self.max_messages]
        # Only summarize user/assistant text turns, not tool traffic.
        text_turns = [m for m in overflow if m.role in ("user", "assistant") and m.content]
        if text_turns:
            digest = "；".join(f"{m.role}: {m.content[:60]}" for m in text_turns[-4:])
            self.summary = (self.summary + " | " + digest)[-self.max_summary_chars :]
        self.messages = self.messages[len(self.messages) - self.max_messages :]

    def as_context(self, system_prompt: str) -> List[Dict[str, Any]]:
        ctx: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if self.summary:
            ctx.append(
                {
                    "role": "system",
                    "content": f"[以下是较早的对话摘要，供参考]\n{self.summary}",
                }
            )
        ctx.extend(m.to_dict() for m in self.messages)
        return ctx

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""
