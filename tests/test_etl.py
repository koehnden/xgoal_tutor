from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xgoal_tutor import etl


@pytest.fixture()
def sample_events(tmp_path: Path) -> Path:
    events_dir = tmp_path / "events"
    lineups_dir = tmp_path / "lineups"
    events_dir.mkdir()
    lineups_dir.mkdir()

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
            "competition": {"id": 1, "name": "Friendly Cup"},
            "season": {"id": 2023, "name": "2023/2024"},
            "match_date": "2023-08-01",
            "venue": "Test Stadium",
            "home_team": {"id": 100, "name": "Team A"},
            "away_team": {"id": 200, "name": "Team B"},
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
    path = events_dir / "match.json"
    path.write_text(json.dumps(events), encoding="utf-8")

    lineups = [
        {
            "team_id": 100,
            "team_name": "Team A",
            "lineup": [
                {
                    "player_id": 10,
                    "player_name": "Playmaker",
                    "jersey_number": 8,
                    "positions": [
                        {
                            "position_id": 8,
                            "position": "Central Midfield",
                            "from": "00:00",
                            "to": "60:00",
                            "from_period": 1,
                            "to_period": 2,
                            "start_reason": "Starting XI",
                        }
                    ],
                },
                {
                    "player_id": 9,
                    "player_name": "Striker",
                    "jersey_number": 9,
                    "positions": [
                        {
                            "position_id": 9,
                            "position": "Striker",
                            "from": "00:00",
                            "to": "90:00",
                            "from_period": 1,
                            "to_period": 2,
                            "start_reason": "Starting XI",
                        }
                    ],
                },
                {
                    "player_id": 20,
                    "player_name": "Impact Sub",
                    "jersey_number": 20,
                    "positions": [
                        {
                            "position_id": 15,
                            "position": "Winger",
                            "from": "60:00",
                            "to": "90:00",
                            "from_period": 2,
                            "to_period": 2,
                            "start_reason": "Substitution",
                        }
                    ],
                },
            ],
        },
        {
            "team_id": 200,
            "team_name": "Team B",
            "lineup": [
                {
                    "player_id": 30,
                    "player_name": "Goalkeeper",
                    "jersey_number": 1,
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
                },
                {
                    "player_id": 40,
                    "player_name": "Defender",
                    "jersey_number": 5,
                    "positions": [
                        {
                            "position_id": 2,
                            "position": "Centre Back",
                            "from": "00:00",
                            "to": "90:00",
                            "from_period": 1,
                            "to_period": 2,
                            "start_reason": "Starting XI",
                        }
                    ],
                },
            ],
        },
    ]

    (lineups_dir / "match.json").write_text(json.dumps(lineups), encoding="utf-8")
    return path


def test_load_match_events_populates_sqlite(sample_events: Path, tmp_path: Path) -> None:
    database = tmp_path / "shots.db"
    etl.load_match_events(sample_events, database)

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        events = conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()
        assert events["cnt"] == 3

        teams = conn.execute(
            "SELECT team_id, team_name FROM teams ORDER BY team_id"
        ).fetchall()
        assert [(row["team_id"], row["team_name"]) for row in teams] == [
            (100, "Team A"),
            (200, "Team B"),
        ]

        players = conn.execute("SELECT COUNT(*) AS cnt FROM players").fetchone()
        assert players["cnt"] == 7

        match = conn.execute(
            "SELECT * FROM matches WHERE match_id = ?",
            (1,),
        ).fetchone()
        assert match is not None
        assert match["home_team_id"] == 100
        assert match["away_team_id"] == 200
        assert match["competition_name"] == "Friendly Cup"
        assert match["season_name"] == "2023/2024"
        assert match["match_date"] == "2023-08-01"
        assert match["venue"] == "Test Stadium"
        assert match["match_label"] == "Team A vs Team B"

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
        assert open_play_shot["is_goal"] == 1
        assert open_play_shot["score_home"] == 1
        assert open_play_shot["score_away"] == 0

        free_kick_shot = shots[1]
        assert free_kick_shot["is_set_piece"] == 1
        assert free_kick_shot["is_free_kick"] == 1
        assert free_kick_shot["assist_type"] is None
        assert free_kick_shot["is_goal"] == 0
        assert free_kick_shot["score_home"] == 1
        assert free_kick_shot["score_away"] == 0

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

        lineups = conn.execute(
            """
            SELECT player_id, team_id, is_starter, sort_order, from_period, from_minute, raw_json
            FROM match_lineups
            ORDER BY team_id, sort_order IS NULL, sort_order, player_id
            """
        ).fetchall()

        assert len(lineups) == 5

        starters = [row for row in lineups if row["is_starter"] == 1]
        assert [(row["team_id"], row["player_id"], row["sort_order"]) for row in starters] == [
            (100, 10, 1),
            (100, 9, 2),
            (200, 30, 1),
            (200, 40, 2),
        ]

        subs = [row for row in lineups if row["is_starter"] == 0]
        assert len(subs) == 1
        sub = subs[0]
        assert sub["player_id"] == 20
        assert sub["from_period"] == 2
        assert sub["from_minute"] == 60
        assert sub["raw_json"] is not None
        raw_event = json.loads(stored_event["raw_json"])
        assert raw_event["id"] == "shot-1"


def test_load_match_events_uses_filename_for_match_id(tmp_path: Path) -> None:
    events = [
        {
            "id": "event-1",
            "type": {"name": "Pass"},
        }
    ]

    events_path = tmp_path / "987654.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    database = tmp_path / "match_id.sqlite"

    etl.load_match_events(events_path, database)

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        stored = conn.execute(
            "SELECT match_id FROM events WHERE event_id = ?", ("event-1",)
        ).fetchone()

        assert stored is not None
        assert stored["match_id"] == 987654

        match = conn.execute(
            "SELECT match_id FROM matches WHERE match_id = ?",
            (987654,),
        ).fetchone()
        assert match is not None


def test_main_invokes_loader(monkeypatch: pytest.MonkeyPatch, sample_events: Path, tmp_path: Path) -> None:
    database = tmp_path / "cli.sqlite"
    called: dict[str, tuple[Path, Path]] = {}

    def fake_loader(events_path: Path, db_path: Path) -> None:
        called["args"] = (events_path, db_path)

    monkeypatch.setattr(etl, "load_match_events", fake_loader)

    etl.main([str(sample_events), str(database)])

    assert called["args"] == (sample_events, database)
