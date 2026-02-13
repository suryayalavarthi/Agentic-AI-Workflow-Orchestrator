from __future__ import annotations

from src.config import Settings, get_settings


def test_defaults_load():
    settings = Settings(anthropic_api_key="k")
    assert settings.default_model == "claude-3-haiku-20240307"
    assert settings.default_temperature == 0.0
    assert settings.max_loop_count == 15
    assert settings.max_context_messages == 6
    assert settings.recursion_limit == 25
    assert settings.chroma_path == "./data/chroma"


def test_env_override(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("src.config._settings", None)
    settings = Settings()
    assert settings.default_model == "claude-3-5-sonnet-20241022"


def test_get_settings_singleton(mock_settings):
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
