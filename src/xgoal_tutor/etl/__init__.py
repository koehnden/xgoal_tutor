from __future__ import annotations

from typing import Optional, Sequence

from xgoal_tutor.etl import statsbomb as _statsbomb

load_match_events = _statsbomb.load_match_events
prepare_metadata_cache = _statsbomb.prepare_metadata_cache


def main(argv: Optional[Sequence[str]] = None) -> None:
    _statsbomb.main(argv, loader=load_match_events)

__all__ = ["load_match_events", "prepare_metadata_cache", "main"]
