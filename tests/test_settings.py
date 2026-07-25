"""Tests for the model API-key settings path (Tauri desktop Phase 2).

A Tauri-launched sidecar doesn't inherit the shell env, so the key may live only in the
SecretStore. These cover: the env→store resolver, the status shape (never leaks the key),
and the REST round-trip. No network, no model calls.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from coworker.providers import resolve_api_key
from coworker.secrets import SecretStore


def test_resolve_api_key_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-123")
    secrets = SecretStore(path=tmp_path / "secrets.json")
    secrets.put("provider:openai", {"type": "api_key", "api_key": "sk-store-999"})
    assert resolve_api_key(secrets) == "sk-env-123"


def test_resolve_api_key_falls_back_to_store(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    secrets = SecretStore(path=tmp_path / "secrets.json")
    assert resolve_api_key(secrets) is None
    secrets.put("provider:openai", {"type": "api_key", "api_key": "sk-store-999"})
    assert resolve_api_key(secrets) == "sk-store-999"


def test_settings_rest_roundtrip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    client = TestClient(create_app(manager))

    before = client.get("/v1/settings").json()
    assert (
        before["has_key"] is False
        and before["source"] is None
        and before["provider"] == "openai"
    )
    assert before["onboarded"] is False and before["model"] in before["models"]

    set_resp = client.post(
        "/v1/settings/model-key", json={"api_key": "sk-secret-xyz"}
    ).json()
    assert (
        set_resp["ok"] is True
        and set_resp["has_key"] is True
        and set_resp["source"] == "store"
    )

    after = client.get("/v1/settings").json()
    assert after["has_key"] is True
    # the key value is never returned by either endpoint
    assert "sk-secret-xyz" not in str(set_resp) and "api_key" not in after

    # empty key is rejected
    assert (
        client.post("/v1/settings/model-key", json={"api_key": "  "}).json()["ok"]
        is False
    )


def test_default_model_and_onboarding_persist(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    data_dir = tmp_path / "data"
    client = TestClient(create_app(SessionManager(data_dir=data_dir)))

    # set a default model + mark onboarded
    assert (
        client.post("/v1/settings/default-model", json={"model": "gpt-4o"}).json()[
            "model"
        ]
        == "gpt-4o"
    )
    assert (
        client.post("/v1/settings/onboarded", json={"value": True}).json()["onboarded"]
        is True
    )
    assert (
        client.post("/v1/settings/default-model", json={"model": " "}).json()["ok"]
        is False
    )

    # a fresh manager over the same data dir restores both from prefs.json
    reborn = SessionManager(data_dir=data_dir)
    assert reborn.model == "gpt-4o"
    s = reborn.get_settings()
    assert s["onboarded"] is True and s["model"] == "gpt-4o"


def test_nav_layout_setting_roundtrips(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    data_dir = tmp_path / "data"
    client = TestClient(create_app(SessionManager(data_dir=data_dir)))

    # defaults to "flat"
    assert client.get("/v1/settings").json()["nav_layout"] == "flat"

    resp = client.post("/v1/settings/nav-layout", json={"nav_layout": "grouped"}).json()
    assert resp == {"ok": True, "nav_layout": "grouped"}
    assert client.get("/v1/settings").json()["nav_layout"] == "grouped"

    # unknown value falls back to flat; persists across a restart
    assert (
        client.post("/v1/settings/nav-layout", json={"nav_layout": "bogus"}).json()[
            "nav_layout"
        ]
        == "flat"
    )
    client.post("/v1/settings/nav-layout", json={"nav_layout": "grouped"})
    reborn = SessionManager(data_dir=data_dir)
    assert reborn.get_settings()["nav_layout"] == "grouped"


def test_scratch_base_setting_persists_and_drives_provisioning(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    data_dir = tmp_path / "data"
    client = TestClient(create_app(SessionManager(data_dir=data_dir)))

    # defaults to ~/OpenWorker
    assert client.get("/v1/settings").json()["scratch_base"] == "~/OpenWorker"

    base = tmp_path / "my coworker files"
    resp = client.post("/v1/settings/scratch-base", json={"path": str(base)}).json()
    assert resp["ok"] is True and resp["scratch_base"] == str(base)
    assert base.is_dir()  # created on set
    assert (
        client.post("/v1/settings/scratch-base", json={"path": " "}).json()["ok"]
        is False
    )

    # persists across a restart and actually drives where scratch dirs are provisioned
    reborn = SessionManager(data_dir=data_dir)
    assert reborn.get_settings()["scratch_base"] == str(base)
    scratch = reborn._provision_scratch("sess-xyz")
    assert Path(scratch) == (base / "sess-xyz").resolve() and Path(scratch).is_dir()


def test_ollama_models_gated_on_liveness(tmp_path, monkeypatch):
    """`ollama:*` entries show only while a local Ollama answers — keyless must not mean
    always-present (a stray ollama:<junk> pref would otherwise render forever)."""
    from coworker.server.manager import SessionManager

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    manager.add_model("ollama:llama3.3")

    monkeypatch.setattr(SessionManager, "_ollama_alive", lambda self: False)
    assert "ollama:llama3.3" not in manager.get_settings()["models"]

    monkeypatch.setattr(SessionManager, "_ollama_alive", lambda self: True)
    assert "ollama:llama3.3" in manager.get_settings()["models"]


# -- #97: discovering models on an Ollama-class server (auth optional, /api/tags optional) --
def _patch_tag_responses(monkeypatch, routes):
    """Route URL -> (status, json). `routes` values may raise to simulate a dead server."""
    seen: list[tuple[str, dict]] = []

    def fake_get(url, **kwargs):
        seen.append((url, kwargs.get("headers") or {}))
        outcome = routes.get(url)
        if outcome is None:
            raise AssertionError(f"unexpected probe: {url}")
        if isinstance(outcome, Exception):
            raise outcome
        status, payload = outcome
        return SimpleNamespace(status_code=status, json=lambda: payload)

    monkeypatch.setattr("httpx.get", fake_get)
    return seen


_TAGS_URL = "http://localhost:11434/api/tags"
_MODELS_URL = "http://localhost:11434/v1/models"
_NATIVE_OK = (200, {"models": [{"name": "qwen3-coder:30b"}, {"noname": 1}]})
_COMPAT_OK = (200, {"data": [{"id": "qwen3-coder:30b"}, {"noid": 1}]})


_BASE = {"base_url": "http://localhost:11434"}


@pytest.mark.parametrize(
    "profile,routes,expected",
    [
        # stock Ollama: native endpoint answers, no key stored, no auth header sent
        (_BASE, {_TAGS_URL: _NATIVE_OK}, ["qwen3-coder:30b"]),
        # a proxied Ollama: same native endpoint, key forwarded
        ({**_BASE, "api_key": "sk-omlx-abc"}, {_TAGS_URL: _NATIVE_OK}, ["qwen3-coder:30b"]),
        # oMLX: no native API at all — fall back to the OpenAI-compatible listing
        (
            {**_BASE, "api_key": "sk-omlx-abc"},
            {_TAGS_URL: (404, {}), _MODELS_URL: _COMPAT_OK},
            ["qwen3-coder:30b"],
        ),
        # a `/v1` base URL is normalised back to the root before probing
        ({"base_url": "http://localhost:11434/v1"}, {_TAGS_URL: _NATIVE_OK}, ["qwen3-coder:30b"]),
        # rejected, unreachable, or a broken fallback → unknown, never a partial list
        (_BASE, {_TAGS_URL: (401, {})}, None),
        (_BASE, {_TAGS_URL: (500, {})}, None),
        (_BASE, {_TAGS_URL: (404, {}), _MODELS_URL: (401, {})}, None),
        (_BASE, {_TAGS_URL: ConnectionError("refused")}, None),
    ],
)
def test_ollama_tags_auth_and_v1_fallback(tmp_path, monkeypatch, profile, routes, expected):
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    manager.secrets.put("provider:ollama", profile)
    seen = _patch_tag_responses(monkeypatch, routes)

    assert manager._ollama_tags(1.0) == expected
    auth = seen[0][1].get("Authorization")
    assert auth == (f"Bearer {profile['api_key']}" if profile.get("api_key") else None)
    # and the public surfaces built on it agree
    assert manager._ollama_models() == [f"ollama:{m}" for m in (expected or [])]


def test_ollama_models_empty_until_configured(tmp_path, monkeypatch):
    """No stored profile → no probe and no suggestions (unchanged by the auth support)."""
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    _patch_tag_responses(monkeypatch, {})  # any probe at all is a failure
    assert manager._ollama_models() == []


def test_saving_a_key_reprobes_a_stale_liveness_verdict(tmp_path, monkeypatch):
    """A cached "not alive" from an unauthenticated probe must not hide the models for 30s
    after the user saves a working key."""
    import time

    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    # another provider owns the default, so the assertion below can only be satisfied by the
    # liveness probe re-running — not by get_settings() force-keeping the default model.
    manager.secrets.put("provider:openai", {"api_key": "sk-test"})
    manager.add_model("ollama:qwen3-coder:30b")
    manager._ollama_alive_cache = (time.monotonic(), False)
    assert "ollama:qwen3-coder:30b" not in manager.get_settings()["models"]

    _patch_tag_responses(monkeypatch, {_TAGS_URL: _NATIVE_OK})
    manager.set_provider("ollama", {"api_key": "sk-omlx-abc"})
    assert "ollama:qwen3-coder:30b" in manager.get_settings()["models"]
