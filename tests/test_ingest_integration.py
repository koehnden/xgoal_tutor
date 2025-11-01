from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_ingest_cli_populates_lineups(tmp_path: Path, ingest_cli_module) -> None:
    root = tmp_path / "statsbomb"
    events_dir = root / "events"
    lineups_dir = root / "lineups"
    events_dir.mkdir(parents=True)
    lineups_dir.mkdir()

    for match_id in range(1, 13):
        event = [
            {
                "id": f"event-{match_id}",
                "type": {"name": "Pass"},
                "team": {"id": match_id, "name": f"Team {match_id}"},
                "player": {"id": match_id * 10, "name": f"Player {match_id}"},
                "match_id": match_id,
            }
        ]
        (events_dir / f"{match_id}.json").write_text(
            json.dumps(event), encoding="utf-8"
        )

        lineup = [
            {
                "team_id": match_id,
                "team_name": f"Team {match_id}",
                "lineup": [
                    {
                        "player_id": match_id * 10,
                        "player_name": f"Player {match_id}",
                        "jersey_number": match_id,
                        "positions": [
                            {
                                "position_id": 1,
                                "position": "Goalkeeper",
                                "from": "00:00",
                                "to": "90:00",
                                "from_period": 1,
                                "to_period": 2,
                                "start_reason": "Starting XI",
                            }
                        ],
                    }
                ],
            }
        ]
        (lineups_dir / f"{match_id}.json").write_text(
            json.dumps(lineup), encoding="utf-8"
        )

    db_path = tmp_path / "integration.sqlite"
    ingest_cli_module.main(
        [str(events_dir), "--database", str(db_path), "--limit", "10"]
    )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM match_lineups WHERE player_id IS NOT NULL"
        ).fetchone()[0]

    assert count > 0
