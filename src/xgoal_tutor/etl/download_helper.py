from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List
from urllib.request import Request, urlopen

try:  # pragma: no cover - exercised in integration scenarios
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm isn't available
    class _DummyTqdm:
        def __init__(self, iterable=None, **_kwargs):
            self._iterable = iterable

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, *_args, **_kwargs):
            return None

    def tqdm(*args, **kwargs):
        return _DummyTqdm(*args, **kwargs)

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


def materialize_github_subtrees(
    owner: str,
    repo: str,
    ref: str,
    subpaths: Iterable[str],
    dest_root: Path,
) -> List[Path]:
    """Download a GitHub repository ZIP archive and extract specific subpaths.

    Parameters
    ----------
    owner, repo, ref:
        Identify the repository and revision to download.
    subpaths:
        Iterable of directory prefixes (relative to the repo root) that should be
        extracted into ``dest_root``.
    dest_root:
        Directory where the extracted files should be written. The function
        creates directories as needed and only writes JSON blobs.

    Returns
    -------
    List[Path]
        The local directories corresponding to each requested subpath, in the
        same order as provided.
    """

    dest_root.mkdir(parents=True, exist_ok=True)

    normalized = [sp.strip("/") for sp in subpaths]
    prefix_map = {sp: f"{repo}-{ref}/{sp}/" for sp in normalized}
    created: dict[str, Path] = {}

    for sp in normalized:
        (dest_root / sp).mkdir(parents=True, exist_ok=True)

    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{ref}"
    data = _get_bytes(zip_url)
    zf = zipfile.ZipFile(io.BytesIO(data))

    for name in zf.namelist():
        if not name.endswith(".json"):
            continue

        for key, prefix in prefix_map.items():
            if not name.startswith(prefix):
                continue

            relative = name[len(prefix) :].strip("/")
            if not relative:
                continue

            target_dir = dest_root / key
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            created.setdefault(key, target_dir)
            break

    return [created.get(sp, dest_root / sp) for sp in normalized]


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
    "materialize_github_subtrees",
]

