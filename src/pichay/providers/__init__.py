from pichay.providers.anthropic import AnthropicAdapter
from pichay.providers.openai import OpenAIAdapter


def adapters() -> dict[str, object]:
    return {
        "anthropic": AnthropicAdapter(),
        "openai": OpenAIAdapter(),
    }
