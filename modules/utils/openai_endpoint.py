import os
from urllib.parse import urlsplit


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def get_openai_base_url() -> str:
    """Return the configured OpenAI-compatible API base URL."""
    base_url = (os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must be an absolute HTTP(S) URL")
    return base_url


def get_openai_chat_completions_url() -> str:
    """Build the Chat Completions URL without duplicating its path."""
    base_url = get_openai_base_url()
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def get_openai_model(selected_model: str | None, default_model: str = "gpt-5.4-mini") -> str:
    """Use OPENAI_MODEL as an explicit override, otherwise keep the selected model."""
    configured_model = os.environ.get("OPENAI_MODEL", "").strip()
    return configured_model or (selected_model or "").strip() or default_model
