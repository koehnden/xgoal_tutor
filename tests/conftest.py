import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def ingest_cli_module() -> types.ModuleType:
    if "tqdm" not in sys.modules:
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

        sys.modules["tqdm"] = types.SimpleNamespace(
            tqdm=lambda *args, **kwargs: _DummyTqdm(*args, **kwargs)
        )

    spec = importlib.util.spec_from_file_location(
        "ingest_statsbomb_cli",
        Path(__file__).resolve().parents[1] / "scripts" / "ingest_statsbomb.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
