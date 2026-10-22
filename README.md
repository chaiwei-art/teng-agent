# teng-agent

商城智能客服 Agent —— 基于工具调用（Function Calling）+ 对话记忆 + 离线评估的完整 Agent 实现。

与 [rag-lite](https://github.com/chaiwei-art/rag-lite)、teng-mall 同属 teng 系列项目。

## 特性

- **ReAct Agent 循环**：LLM 规划 → 调用工具 → 观察结果 → 生成回答
- **OpenAI Function Calling 协议**：工具 schema、tool_calls、tool result 完全兼容 OpenAI 格式，可无缝切换真实模型
- **免密钥演示**：内置规则 MockLLM，无 API Key 也能跑完整流程（规划 + 工具 + 回答）
- **对话记忆**：滑动窗口 + 滚动摘要，支持多轮上下文
- **5 个业务工具**：查订单、查物流、申请退款、搜商品、FAQ 检索
- **离线评估**：10 条 golden case 回归测试，换模型后可验证效果

## 架构

```
用户输入
   │
   ▼
┌─────────────┐    tool_calls    ┌──────────────┐
│  Agent 循环  │ ───────────────→ │  ToolRegistry │
│  (LLM 规划) │ ←─────────────── │  执行工具     │
└──────┬──────┘    tool result   └──────┬───────┘
       │                                │
       ▼                                ▼
  最终回答（中文）              shop_db.json（mock 数据）
       ▲
       │
┌──────┴──────┐
│  Memory     │  滑动窗口 + 摘要
└─────────────┘
```

## 快速开始

```bash
# 免安装直接运行演示（MockLLM，无需任何依赖）
python examples/demo.py

# 交互式 CLI
python -m teng_agent.cli

# 带工具调用轨迹
python -m teng_agent.cli --debug
```

### 接入真实 LLM

```bash
pip install openai
export OPENAI_API_KEY=sk-...
# 可选：自定义兼容端点（Ollama / vLLM / 中转）
export OPENAI_BASE_URL=https://api.openai.com/v1

python -m teng_agent.cli --model gpt-4o-mini
# 或
python -m teng_agent.cli --model qwen2.5 --api-base http://localhost:11434/v1
```

### 代码调用

```python
from teng_agent import build_agent

agent = build_agent()               # 默认 MockLLM
reply = agent.chat("查一下 TM20261001001 的物流")
print(reply.answer)                 # 中文回答
print(reply.tools_used)             # ['query_logistics']
print(reply.trace)                  # 工具调用轨迹

agent.reset()                       # 清空会话
```

## 试一试这些问题

| 输入 | 触发工具 |
|------|----------|
| 我的订单有哪些 | `query_order` |
| 查一下 TM20261001001 的物流 | `query_logistics` |
| 我的快递到哪了（不带单号） | `query_order` → 列订单引导补单号 |
| 帮我退款 TM20260920002，质量有问题 | `apply_refund` |
| 有没有耳机推荐 | `search_product` |
| 运费多少 | `search_faq` |

## 测试与评估

```bash
# 单元测试（20+ 用例）
python -m pytest tests/ -v
# 或
python -m unittest discover tests

# 离线评估（10 条 golden case）
python eval/run_eval.py
```

## 扩展指南

**加工具**：在 `teng_agent/tools/` 新建模块，用 `@registry.tool(...)` 装饰函数即可：

```python
@registry.tool(
    name="query_coupon",
    description="查询用户可用优惠券",
    parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
)
def query_coupon(user_id: str):
    return {"可用优惠券": []}
```

**换数据**：`--db path/to/shop_db.json` 指定自定义 mock 数据。

**加评估**：在 `eval/run_eval.py` 的 `CASES` 中追加 golden case。

## 项目结构

```
teng-agent/
├── teng_agent/
│   ├── agent.py           # ReAct 核心循环
│   ├── memory.py          # 对话记忆（滑窗 + 摘要）
│   ├── cli.py             # 交互式命令行
│   ├── tools/
│   │   ├── base.py        # Tool / ToolRegistry / ToolCall
│   │   └── shop.py        # 5 个商城业务工具
│   ├── llm/
│   │   ├── base.py        # LLM 抽象接口
│   │   ├── openai_llm.py  # OpenAI 兼容实现
│   │   └── mock_llm.py    # 规则 Mock（免密钥演示）
│   └── mock_data/
│       └── shop_db.json   # 订单 / 商品 / FAQ
├── examples/demo.py       # 非交互演示
├── eval/run_eval.py       # 离线评估
└── tests/                 # 单元测试
```

## License

MIT
