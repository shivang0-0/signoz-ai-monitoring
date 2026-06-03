from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


def get_llm() -> BaseChatModel:
    model = os.environ.get("LLM_MODEL", "qwen2.5-coder:1.5b")
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=SecretStr("sk-ollama"),
        base_url=base_url,
    )
