from __future__ import annotations

import contextlib
import json
import os
import shutil
import ssl
import subprocess
from typing import Optional
from urllib.request import Request, urlopen


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi’s CA bundle; fall back to system defaults."""

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _headers(extra: Optional[dict] = None) -> dict:
    hdrs = {"User-Agent": "xgoal-tutor/ingest"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if extra:
        hdrs.update(extra)
    return hdrs


def _get_bytes(url: str) -> bytes:
    """GET bytes with urllib; on error, try curl once (helps on macOS)."""

    req = Request(url, headers=_headers())
    try:
        with contextlib.closing(urlopen(req, context=_ssl_context(), timeout=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            return resp.read()
    except Exception:
        curl = shutil.which("curl")
        if curl:
            try:
                return subprocess.check_output(
                    [curl, "-fsSL", "--retry", "3", url],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError:
                pass
        raise


def _get_json(url: str):
    data = _get_bytes(url)
    return json.loads(data.decode("utf-8"))


__all__ = [
    "_get_bytes",
    "_get_json",
    "_headers",
    "_ssl_context",
]

