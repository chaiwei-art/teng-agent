from teng_agent.llm.base import BaseLLM, LLMResponse
from teng_agent.llm.mock_llm import MockLLM

__all__ = ["BaseLLM", "LLMResponse", "MockLLM"]


def __getattr__(name: str):
    if name == "OpenAILLM":
        from teng_agent.llm.openai_llm import OpenAILLM

        return OpenAILLM
    raise AttributeError(name)
