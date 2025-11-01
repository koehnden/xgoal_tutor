from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import tarfile
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator
from urllib.request import Request, urlopen

from tqdm import tqdm

from xgoal_tutor.etl.http_helper import _get_bytes, _get_json, _headers, _ssl_context


logger = logging.getLogger(__name__)


def _download_to_tempfile(url: str) -> Path:
    raw = _get_bytes(url)
    tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(raw)
        tmp.flush()
    finally:
        tmp.close()
    data = json.loads(tmp_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from {url}")
    return tmp_path


def _raw_url(owner: str, repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path.lstrip('/')}"


def _list_events_with_trees_api(owner: str, repo: str, ref: str, subpath: str) -> list[str]:
    """Git Trees API (recursive) → list all blob paths; filter to subpath/*.json."""

    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    tree = _get_json(api)
    if not isinstance(tree, dict) or "tree" not in tree:
        raise RuntimeError(f"Unexpected response from Git Trees API: {api}")

    base = subpath.strip("/").rstrip("/") + "/"
    paths = [
        e["path"]
        for e in tree.get("tree", [])
        if e.get("type") == "blob"
        and e.get("path", "").startswith(base)
        and e["path"].endswith(".json")
    ]

    if tree.get("truncated"):
        return []
    return paths


def download_github_directory_jsons(owner: str, repo: str, ref: str, subpath: str) -> Iterator[Path]:
    """
    Preferred: stream the repository tarball and extract subpath/*.json on the fly.
    Fallbacks: per-file downloads (Trees API) and repository ZIP extraction.
    """

    try:
        yield from _stream_tarball_events(owner, repo, ref, subpath)
        return
    except Exception as e:
        logger.warning(
            "[warn] Tarball streaming failed; trying per-file download: %s",
            e,
            exc_info=e,
        )

    try:
        paths = _list_events_with_trees_api(owner, repo, ref, subpath)
        if paths:
            for rel in tqdm(paths, desc="Downloading events", unit="file"):
                yield _download_to_tempfile(_raw_url(owner, repo, ref, rel))
            return
    except Exception as exc:
        logger.warning("Trees API listing failed; trying ZIP fallback", exc_info=exc)

    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{ref}"
    data = _get_bytes(zip_url)
    zf = zipfile.ZipFile(io.BytesIO(data))
    prefix = f"{repo}-{ref}/" + subpath.strip("/").rstrip("/") + "/"

    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".json")]
    for name in tqdm(names, desc="Extracting events", unit="file"):
        tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
        with tmp:
            tmp.write(zf.read(name))
        yield Path(tmp.name)


def _stream_tarball_events(owner: str, repo: str, ref: str, subpath: str) -> Iterator[Path]:
    tar_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"
    req = Request(tar_url, headers=_headers())
    with contextlib.closing(urlopen(req, context=_ssl_context(), timeout=60)) as resp:
        status = getattr(resp, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP {status} for {tar_url}")

        mode = "r|gz"
        prefix_root = subpath.strip("/")
        prefix = f"{repo}-{ref}/"
        if prefix_root:
            prefix += prefix_root.rstrip("/") + "/"

        with tarfile.open(fileobj=resp, mode=mode) as tf, tqdm(
            desc="Streaming events", unit="file"
        ) as progress:
            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                if not name.startswith(prefix) or not name.endswith(".json"):
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
                with tmp, contextlib.closing(extracted):
                    shutil.copyfileobj(extracted, tmp)

                progress.update(1)
                yield Path(tmp.name)


__all__ = [
    "download_github_directory_jsons",
    "_download_to_tempfile",
    "_list_events_with_trees_api",
    "_raw_url",
    "_stream_tarball_events",
]

