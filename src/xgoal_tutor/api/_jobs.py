"""SQLite-backed job registry for asynchronous summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from xgoal_tutor.api._database import get_connection
from xgoal_tutor.api.models import JobRecord

_JOBS_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS summary_jobs (
      generation_id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      match_id TEXT NOT NULL,
      player_id TEXT,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      expires_at TEXT,
      result_json TEXT,
      error_message TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_match ON summary_jobs(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_player ON summary_jobs(player_id);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_kind ON summary_jobs(kind);",
)

_PREDICTION_CACHE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS match_prediction_cache (
      match_id TEXT PRIMARY KEY,
      response_json TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
    """,
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    for statement in (*_JOBS_SCHEMA, *_PREDICTION_CACHE_SCHEMA):
        conn.execute(statement)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def insert_job(
    generation_id: str,
    *,
    kind: str,
    match_id: str,
    player_id: Optional[str] = None,
    status: str = "queued",
) -> JobRecord:
    """Persist a new job in the registry and return its record."""

    created = _format_timestamp(_utcnow())
    with get_connection() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO summary_jobs (
                generation_id, kind, match_id, player_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (generation_id, kind, match_id, player_id, status, created, created),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM summary_jobs WHERE generation_id = ?", (generation_id,)
        ).fetchone()
    return _row_to_job(row)


def update_status(
    generation_id: str,
    status: str,
    *,
    error_message: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> None:
    """Update the status (and optionally error) for a job."""

    timestamp = _format_timestamp(_utcnow())
    with get_connection() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            UPDATE summary_jobs
               SET status = ?, updated_at = ?, error_message = ?, expires_at = ?
             WHERE generation_id = ?
            """,
            (status, timestamp, error_message, expires_at, generation_id),
        )
        conn.commit()


def store_result(
    generation_id: str,
    result: Dict[str, Any],
    *,
    status: str = "done",
    expires_at: Optional[str] = None,
) -> None:
    """Store the final result payload for a job and mark its status."""

    timestamp = _format_timestamp(_utcnow())
    payload = json.dumps(result)
    with get_connection() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            UPDATE summary_jobs
               SET status = ?, result_json = ?, updated_at = ?, expires_at = ?, error_message = NULL
             WHERE generation_id = ?
            """,
            (status, payload, timestamp, expires_at, generation_id),
        )
        conn.commit()


def get_job(generation_id: str) -> Optional[JobRecord]:
    """Fetch a job by identifier."""

    with get_connection() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM summary_jobs WHERE generation_id = ?", (generation_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def get_latest_job(kind: str, match_id: str, player_id: Optional[str] = None) -> Optional[JobRecord]:
    """Return the latest job for the given identifiers."""

    query = [
        "SELECT * FROM summary_jobs WHERE kind = ? AND match_id = ?",
    ]
    params: list[Any] = [kind, match_id]
    if player_id is not None:
        query.append("AND player_id = ?")
        params.append(player_id)
    query.append("ORDER BY datetime(created_at) DESC LIMIT 1")
    sql = " ".join(query)

    with get_connection() as conn:
        _ensure_schema(conn)
        row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def cache_match_predictions(match_id: str, response: Dict[str, Any]) -> None:
    """Persist predictions for a match so child jobs can reuse them."""

    timestamp = _format_timestamp(_utcnow())
    payload = json.dumps(response)
    with get_connection() as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO match_prediction_cache (match_id, response_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
                response_json = excluded.response_json,
                updated_at = excluded.updated_at
            """,
            (match_id, payload, timestamp),
        )
        conn.commit()


def get_cached_match_predictions(match_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached predictions for a match, if present."""

    with get_connection() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT response_json FROM match_prediction_cache WHERE match_id = ?",
            (match_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["response_json"])


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    result_json = row["result_json"]
    result = json.loads(result_json) if result_json else None
    return JobRecord(
        generation_id=row["generation_id"],
        kind=row["kind"],
        match_id=row["match_id"],
        player_id=row["player_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
        result=result,
        error_message=row["error_message"],
    )
