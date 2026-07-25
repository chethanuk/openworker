"""Authenticated local OpenAI-compatible servers behind the Ollama provider slot (#97).

Some local servers ask for an API key — oMLX on a Mac, or a plain `ollama serve` behind an auth
proxy. These drive the same REST surface the GUI uses (`/v1/providers*`, `/v1/settings`) against a
real in-process stub server, so the whole path is covered: the descriptor the form renders from,
the Test/Detect probe, model discovery, and the key that rides on actual model requests.

The stub mimics oMLX: it speaks ONLY the OpenAI-compatible `/v1` API (no native `/api/tags`) and
rejects any request without `Authorization: Bearer <key>`. No network — an ephemeral localhost port,
the same in-process-harness pattern `coworker.testing.fake_slack` uses.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import pytest

STUB_KEY = "sk-omlx-test"
STUB_MODELS = ["qwen3-coder:30b", "mlx-community/Qwen3-8B"]


class _LocalServer:
    """An oMLX-like local server: OpenAI-compatible `/v1` only, Bearer-authenticated.

    `require_key=False` makes it behave like a stock `ollama serve` (auth-free) so the same
    harness covers the keyless case. `auth_headers` records what each request presented.
    """

    def __init__(self, *, require_key: bool = True, native_tags: bool = False) -> None:
        self.require_key = require_key
        self.native_tags = native_tags
        self.auth_headers: list[Optional[str]] = []
        self.paths: list[str] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep pytest output clean
                pass

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authed(self) -> bool:
                header = self.headers.get("Authorization")
                stub.auth_headers.append(header)
                stub.paths.append(self.path)
                if not stub.require_key:
                    return True
                return header == f"Bearer {STUB_KEY}"

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
                if self.path.rstrip("/") == "/api/tags" and not stub.native_tags:
                    stub.paths.append(self.path)
                    self._send(404, {"detail": "Not Found"})
                    return
                if not self._authed():
                    self._send(401, {"error": {"message": "Unauthorized"}})
                    return
                if self.path.rstrip("/") == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in STUB_MODELS]})
                    return
                if self.path.rstrip("/") == "/v1/models":
                    self._send(
                        200,
                        {"object": "list", "data": [{"id": m} for m in STUB_MODELS]},
                    )
                    return
                self._send(404, {"detail": "Not Found"})

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if not self._authed():
                    self._send(401, {"error": {"message": "Unauthorized"}})
                    return
                self._send(
                    200,
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": STUB_MODELS[0],
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "pong"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_LocalServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _client(tmp_path, monkeypatch):
    """The sidecar REST app the GUI talks to, on an isolated state dir."""
    from fastapi.testclient import TestClient

    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data")
    return TestClient(create_app(manager)), manager


# -- the reporter's symptom: Test/Detect against a key-requiring local server ----
@pytest.mark.parametrize(
    "require_key,sent_key,expect_ok,expect_error",
    [
        # The issue: an authenticated local server + the right key must verify.
        (True, STUB_KEY, True, None),
        # Today's behaviour, and the symptom: no key against that server is rejected.
        (True, "", False, "add or check the API key"),
        (True, "sk-wrong", False, "add or check the API key"),
        # A stock auth-free `ollama serve` must keep working, with or without a key.
        (False, "", True, None),
        (False, STUB_KEY, True, None),
    ],
)
def test_verify_ollama_through_rest(
    tmp_path, monkeypatch, require_key, sent_key, expect_ok, expect_error
):
    client, _ = _client(tmp_path, monkeypatch)
    with _LocalServer(require_key=require_key) as server:
        res = client.post(
            "/v1/providers/verify",
            json={
                "name": "ollama",
                "fields": {"base_url": server.base_url, "api_key": sent_key},
            },
        ).json()
    assert res["ok"] is expect_ok, res
    if expect_error:
        assert expect_error in res["error"]


# -- the form the GUI renders: an optional, secret api_key below the server URL ---
def test_providers_endpoint_exposes_optional_api_key(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    ollama = next(p for p in client.get("/v1/providers").json() if p["name"] == "ollama")

    keys = [f["key"] for f in ollama["fields"]]
    assert keys == ["base_url", "api_key"], "the key field renders below the server URL"
    key_field = ollama["fields"][1]
    assert key_field["secret"] is True and key_field["required"] is False
    # Keyless-by-default: the provider stays usable with the field left empty.
    assert ollama["needs_key"] is False and ollama["configured"] is True
    assert ollama["has_key"] is False


@pytest.mark.parametrize("native_tags", [False, True], ids=["omlx", "ollama"])
def test_saved_key_round_trips_and_clears(tmp_path, monkeypatch, native_tags):
    """Saving, reading back, and clearing the optional key — never leaking its value."""
    client, _ = _client(tmp_path, monkeypatch)
    with _LocalServer(native_tags=native_tags) as server:
        saved = client.post(
            "/v1/providers",
            json={
                "name": "ollama",
                "fields": {"base_url": server.base_url, "api_key": STUB_KEY},
            },
        ).json()
        assert saved["ok"] is True

        ollama = next(
            p for p in client.get("/v1/providers").json() if p["name"] == "ollama"
        )
        assert ollama["has_key"] is True
        assert ollama["values"]["base_url"] == server.base_url
        assert "api_key" not in ollama["values"], "a secret never goes back to the GUI"
        assert ollama["key_set_at"]

        # Clearing the key keeps the endpoint (the GUI's "Remove key…" on a keyless provider).
        cleared = client.post(
            "/v1/providers", json={"name": "ollama", "fields": {"api_key": ""}}
        ).json()
        assert cleared["ok"] is True
        ollama = next(
            p for p in client.get("/v1/providers").json() if p["name"] == "ollama"
        )
        assert ollama["has_key"] is False
        assert ollama["values"]["base_url"] == server.base_url
        assert ollama["key_set_at"] is None, "no stamp for a key that no longer exists"


# -- the picker: models must appear for a server that has no native /api/tags -----
@pytest.mark.parametrize("native_tags", [False, True], ids=["omlx", "ollama"])
def test_settings_lists_models_from_authenticated_server(
    tmp_path, monkeypatch, native_tags
):
    client, _ = _client(tmp_path, monkeypatch)
    with _LocalServer(native_tags=native_tags) as server:
        # The GUI reads settings before the provider is configured (nothing local yet).
        assert not [
            m for m in client.get("/v1/settings").json()["models"] if m.startswith("ollama:")
        ]
        client.post(
            "/v1/providers",
            json={
                "name": "ollama",
                "fields": {"base_url": server.base_url, "api_key": STUB_KEY},
            },
        )
        # Immediately after saving — a stale liveness verdict must not hide the models.
        models = client.get("/v1/settings").json()["models"]
        assert f"ollama:{STUB_MODELS[0]}" in models, models

        ollama = next(
            p for p in client.get("/v1/providers").json() if p["name"] == "ollama"
        )
        assert STUB_MODELS[0] in ollama["suggested_models"]


# -- the key must ride on real model requests, not just the Test button -----------
def test_model_requests_send_the_stored_key(tmp_path, monkeypatch):
    from coworker.providers.registry import build_provider_client

    client, manager = _client(tmp_path, monkeypatch)
    with _LocalServer() as server:
        client.post(
            "/v1/providers",
            json={
                "name": "ollama",
                "fields": {"base_url": server.base_url, "api_key": STUB_KEY},
            },
        )
        profile = manager.secrets.get("provider:ollama")
        provider = build_provider_client("ollama", profile, manager.secrets)
        turn = provider.complete(
            model=STUB_MODELS[0], messages=[{"role": "user", "content": "ping"}]
        )
    assert turn.text == "pong"
    assert f"Bearer {STUB_KEY}" in server.auth_headers
