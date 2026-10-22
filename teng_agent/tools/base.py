from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ToolCall:
    tool: str
    arguments: Dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    tool: str
    success: bool
    output: Any
    call_id: str = ""

    def to_text(self) -> str:
        payload = {
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
        }
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any] = field(repr=False, default=None)

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            result = self.func(**kwargs)
            return ToolResult(tool=self.name, success=True, output=result)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool=self.name, success=False, output=str(exc))

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any] | None = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.register(
                Tool(name=name, description=description, parameters=parameters or {"type": "object", "properties": {}}, func=func)
            )
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(tool=call.tool, success=False, output=f"Unknown tool: {call.tool}", call_id=call.call_id)
        result = tool.run(**call.arguments)
        result.call_id = call.call_id
        return result


def infer_parameters(func: Callable) -> Dict[str, Any]:
    """Build a minimal JSON schema from a function signature (best-effort)."""
    sig = inspect.signature(func)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = param.annotation
        ptype = "string"
        if ann in (int,):
            ptype = "integer"
        elif ann in (float,):
            ptype = "number"
        elif ann in (bool,):
            ptype = "boolean"
        props[pname] = {"type": ptype}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}
