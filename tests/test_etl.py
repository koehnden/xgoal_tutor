from __future__ import annotations

import json
import ssl
import sqlite3
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from urllib.error import URLError

from xgoal_tutor import etl
from xgoal_tutor.ingest import loader, reader, rows


@pytest.fixture()
def sample_event_payload() -> list[dict[str, Any]]:
    return [
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
                "outcome": {"id": 90, "name": "Off Target"},
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


@pytest.fixture()
def sample_events(tmp_path: Path, sample_event_payload: list[dict[str, Any]]) -> Path:
    path = tmp_path / "events.json"
    path.write_text(json.dumps(sample_event_payload), encoding="utf-8")
    return path


@pytest.fixture()
def sample_events_list(
    sample_event_payload: list[dict[str, Any]]
) -> list[reader.MutableEvent]:
    normalised = [dict(event) for event in sample_event_payload]
    return normalised


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


def test_main_invokes_loader(
    monkeypatch: pytest.MonkeyPatch, sample_events: Path, tmp_path: Path
) -> None:
    database = tmp_path / "cli.sqlite"
    called: dict[str, tuple[str | Path, Path | str]] = {}

    def fake_loader(events_path: Path | str, db_path: Path | str) -> None:
        called["args"] = (events_path, db_path)

    monkeypatch.setattr(etl, "load_match_events", fake_loader)

    etl.main([str(sample_events), str(database)])

    assert called["args"] == (str(sample_events), database)


def test_read_statsbomb_events_raises_on_ssl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/events.json"

    def fake_urlopen(*_: Any, **__: Any) -> Any:
        raise URLError(ssl.SSLError("certificate verify failed"))

    monkeypatch.setattr(reader, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError) as excinfo:
        reader.read_statsbomb_events(url)

    assert "certificate" in str(excinfo.value).lower()


def test_read_statsbomb_events_falls_back_to_unverified_context(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    url = "https://example.com/events.json"
    attempts: list[int] = []

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b"[]"

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    def fake_urlopen(*_: Any, context: ssl.SSLContext, **__: Any) -> FakeResponse:
        attempts.append(context.verify_mode)
        if context.verify_mode == ssl.CERT_REQUIRED:
            raise URLError("certificate verify failed")
        return FakeResponse()

    monkeypatch.setattr(reader, "urlopen", fake_urlopen)

    events = reader.read_statsbomb_events(url)

    assert events == []
    assert attempts == [ssl.CERT_REQUIRED, ssl.CERT_NONE]
    assert any("unverified" in str(w.message).lower() for w in recwarn)


def test_reader_validates_input(tmp_path: Path) -> None:
    events_file = tmp_path / "invalid.json"
    events_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        reader.read_statsbomb_events(events_file)


def test_reader_rejects_non_json_urls() -> None:
    with pytest.raises(ValueError, match="JSON file or a GitHub directory"):
        reader.read_statsbomb_events("https://example.com/events")


def test_reader_downloads_remote_json(
    tmp_path: Path, sample_event_payload: list[dict[str, Any]]
) -> None:
    events_file = tmp_path / "remote.json"
    events_file.write_text(json.dumps(sample_event_payload), encoding="utf-8")

    class SilentHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - silence server
            return

    handler = partial(SilentHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        url = f"http://{host}:{port}/{events_file.name}"
        events = reader.read_statsbomb_events(url)
    finally:
        server.shutdown()
        thread.join()

    assert len(events) == len(sample_event_payload)


def test_reader_handles_directory(tmp_path: Path, sample_event_payload: list[dict[str, Any]]) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    (events_dir / "match_a.json").write_text(
        json.dumps(sample_event_payload[:1]), encoding="utf-8"
    )
    (events_dir / "match_b.json").write_text(
        json.dumps(sample_event_payload[1:]), encoding="utf-8"
    )

    events = reader.read_statsbomb_events(events_dir)

    assert {event["id"] for event in events} == {"pass-1", "shot-1", "shot-2"}


def test_reader_downloads_github_tree(
    monkeypatch: pytest.MonkeyPatch, sample_event_payload: list[dict[str, Any]]
) -> None:
    payload_a = json.dumps(sample_event_payload[:1])
    payload_b = json.dumps(sample_event_payload[1:])

    captured: dict[str, tuple[str, str, str, str]] = {}

    def fake_fetch(owner: str, repo: str, ref: str, directory: str) -> list[str]:
        captured["args"] = (owner, repo, ref, directory)
        return [payload_a, payload_b]

    monkeypatch.setattr(reader, "_fetch_github_directory_files", fake_fetch)

    events = reader.read_statsbomb_events(
        "https://github.com/statsbomb/open-data/tree/master/data/events"
    )

    assert captured["args"] == ("statsbomb", "open-data", "master", "data/events")
    assert {event["id"] for event in events} == {"pass-1", "shot-1", "shot-2"}


def test_builders_create_expected_rows(sample_events_list: list[reader.MutableEvent]) -> None:
    event_rows = rows.build_event_rows(sample_events_list)
    shot_rows, freeze_frame_rows = rows.build_shot_rows(sample_events_list)

    assert {row[0] for row in event_rows} == {"pass-1", "shot-1", "shot-2"}
    assert len(shot_rows) == 2
    assert len(freeze_frame_rows) == 2

    open_play_shot = next(row for row in shot_rows if row[0] == "shot-1")
    assert open_play_shot[21] == "Through Ball"
    assert open_play_shot[32] == 0
    assert open_play_shot[39] == 1


def test_loader_round_trip(tmp_path: Path, sample_events_list: list[reader.MutableEvent]) -> None:
    database = tmp_path / "round_trip.sqlite"
    loader._write_to_database(sample_events_list, database)

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        stored = conn.execute("SELECT * FROM shots ORDER BY shot_id").fetchall()
        assert {shot["shot_id"] for shot in stored} == {"shot-1", "shot-2"}
