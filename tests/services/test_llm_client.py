"""Unit tests for the Ollama LLM client."""

from __future__ import annotations

import os
from unittest.mock import patch

from langchain_openai import ChatOpenAI


@patch("clients.llm_client.load_dotenv")
def test_get_llm_defaults_to_ollama(_mock_dotenv: object) -> None:
    from clients.llm_client import get_llm

    with patch.dict(os.environ, {}, clear=True):
        llm: ChatOpenAI = get_llm()  # type: ignore[assignment]
        assert llm.model_name == "qwen2.5-coder:1.5b"
        assert "11434" in str(llm.openai_api_base)


@patch("clients.llm_client.load_dotenv")
def test_get_llm_respects_model_env(_mock_dotenv: object) -> None:
    from clients.llm_client import get_llm

    with patch.dict(os.environ, {"LLM_MODEL": "llama3.1"}, clear=True):
        llm: ChatOpenAI = get_llm()  # type: ignore[assignment]
        assert llm.model_name == "llama3.1"


@patch("clients.llm_client.load_dotenv")
def test_get_llm_respects_base_url_env(_mock_dotenv: object) -> None:
    from clients.llm_client import get_llm

    with patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:1234/v1"}, clear=True):
        llm: ChatOpenAI = get_llm()  # type: ignore[assignment]
        assert "1234" in str(llm.openai_api_base)
