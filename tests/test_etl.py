from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xgoal_tutor import etl


@pytest.fixture()
def sample_events(tmp_path: Path) -> Path:
    events = [
        {
            "id": "pass-1",
            "index": 5,
            "period": 1,
            "timestamp": "00:05:12.000",
            "minute": 5,
            "second": 12.0,
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 100, "name": "Team A"},
            "opponent": {"id": 200, "name": "Team B"},
            "player": {"id": 10, "name": "Playmaker"},
            "match_id": 1,
            "possession": 15,
            "possession_team": {"id": 100, "name": "Team A"},
            "location": [80.0, 30.0],
            "pass": {
                "recipient": {"id": 9, "name": "Striker"},
                "length": 20.0,
                "angle": 0.2,
                "height": {"id": 1, "name": "Ground Pass"},
                "type": {"id": 1, "name": "Through Ball"},
            },
        },
        {
            "id": "shot-1",
            "index": 6,
            "period": 1,
            "timestamp": "00:05:15.000",
            "minute": 5,
            "second": 15.0,
            "type": {"id": 16, "name": "Shot"},
            "team": {"id": 100, "name": "Team A"},
            "opponent": {"id": 200, "name": "Team B"},
            "player": {"id": 9, "name": "Striker"},
            "match_id": 1,
            "possession": 15,
            "possession_team": {"id": 100, "name": "Team A"},
            "play_pattern": {"id": 1, "name": "From Counter"},
            "location": [100.0, 40.0],
            "under_pressure": True,
            "shot": {
                "statsbomb_xg": 0.34,
                "key_pass_id": "pass-1",
                "outcome": {"id": 97, "name": "Goal"},
                "body_part": {"id": 40, "name": "Right Foot"},
                "technique": {"id": 1, "name": "Normal"},
                "type": {"id": 62, "name": "Open Play"},
                "first_time": True,
                "one_on_one": False,
                "open_goal": False,
                "follows_dribble": False,
                "deflected": False,
                "rebound": False,
                "end_location": [120.0, 40.0, 1.0],
                "freeze_frame": [
                    {
                        "location": [110.0, 40.0],
                        "teammate": False,
                        "keeper": True,
                        "player": {"id": 30, "name": "Goalkeeper"},
                        "position": {"id": 1, "name": "Goalkeeper"},
                    },
                    {
                        "location": [104.0, 35.0],
                        "teammate": True,
                        "keeper": False,
                        "player": {"id": 11, "name": "Winger"},
                        "position": {"id": 4, "name": "Midfielder"},
                    },
                ],
            },
        },
        {
            "id": "shot-2",
            "index": 10,
            "period": 1,
            "timestamp": "00:20:00.000",
            "minute": 20,
            "second": 0.0,
            "type": {"id": 16, "name": "Shot"},
            "team": {"id": 100, "name": "Team A"},
            "opponent": {"id": 200, "name": "Team B"},
            "player": {"id": 14, "name": "Midfielder"},
            "match_id": 1,
            "possession": 22,
            "possession_team": {"id": 100, "name": "Team A"},
            "play_pattern": {"id": 2, "name": "From Free Kick"},
            "location": [90.0, 20.0],
            "shot": {
                "statsbomb_xg": 0.05,
                "outcome": {"id": 90, "name": "Off T"},
                "body_part": {"id": 40, "name": "Left Foot"},
                "type": {"id": 65, "name": "Free Kick"},
                "first_time": False,
                "one_on_one": False,
                "open_goal": False,
                "follows_dribble": False,
                "deflected": True,
                "end_location": [120.0, 45.0, 0.5],
            },
        },
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")
    return path


def test_load_match_events_populates_sqlite(sample_events: Path, tmp_path: Path) -> None:
    database = tmp_path / "shots.db"
    etl.load_match_events(sample_events, database)

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        events = conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        assert events["cnt"] == 3

        shots = conn.execute(
            "SELECT * FROM shots ORDER BY shot_id"
        ).fetchall()
        assert len(shots) == 2

        open_play_shot = shots[0]
        assert open_play_shot["assist_type"] == "Through Ball"
        assert open_play_shot["is_set_piece"] == 0
        assert open_play_shot["freeze_frame_available"] == 1
        assert open_play_shot["freeze_frame_count"] == 2
        assert pytest.approx(open_play_shot["statsbomb_xg"], rel=1e-5) == 0.34
        assert open_play_shot["start_x"] == 100.0
        assert open_play_shot["end_z"] == 1.0

        free_kick_shot = shots[1]
        assert free_kick_shot["is_set_piece"] == 1
        assert free_kick_shot["is_free_kick"] == 1
        assert free_kick_shot["assist_type"] is None

        freeze_frames = conn.execute(
            "SELECT * FROM freeze_frames WHERE shot_id = ? ORDER BY keeper DESC",
            ("shot-1",),
        ).fetchall()
        assert len(freeze_frames) == 2
        goalkeeper = freeze_frames[0]
        assert goalkeeper["keeper"] == 1
        assert goalkeeper["teammate"] == 0
        assert goalkeeper["player_name"] == "Goalkeeper"
        assert pytest.approx(goalkeeper["x"], rel=1e-5) == 110.0

        stored_event = conn.execute(
            "SELECT raw_json FROM events WHERE event_id = ?", ("shot-1",)
        ).fetchone()
        assert stored_event is not None
        raw_event = json.loads(stored_event["raw_json"])
        assert raw_event["id"] == "shot-1"


def test_main_invokes_loader(monkeypatch: pytest.MonkeyPatch, sample_events: Path, tmp_path: Path) -> None:
    database = tmp_path / "cli.sqlite"
    called: dict[str, tuple[Path, Path]] = {}

    def fake_loader(events_path: Path, db_path: Path) -> None:
        called["args"] = (events_path, db_path)

    monkeypatch.setattr(etl, "load_match_events", fake_loader)

    etl.main([str(sample_events), str(database)])

    assert called["args"] == (sample_events, database)
