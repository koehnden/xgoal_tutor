"""Query helpers for match listing endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import sqlite3
from fastapi import HTTPException

from xgoal_tutor.api._database import get_db
from xgoal_tutor.api._row_utils import row_value, team_payload


def _normalise_kickoff(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    dt: datetime | None
    try:
        dt = datetime.fromisoformat(iso_text)
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return text

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_match_label(
    home_name: str | None,
    away_name: str | None,
    kickoff_utc: str | None,
    fallback: str | None,
) -> str | None:
    if home_name and away_name and kickoff_utc:
        date_part = kickoff_utc.split("T", 1)[0]
        if date_part:
            return f"{home_name} – {away_name} ({date_part})"

    if fallback:
        return fallback

    if home_name and away_name:
        return f"{home_name} – {away_name}"

    return None


def list_matches(page: int, page_size: int) -> Dict[str, Any]:
    offset = (page - 1) * page_size

    with get_db() as connection:
        try:
            cursor = connection.execute("SELECT COUNT(*) AS total FROM matches")
            total_row = cursor.fetchone()
            total = int(total_row["total"]) if total_row and total_row["total"] is not None else 0

            rows: List[sqlite3.Row] = connection.execute(
                """
                SELECT
                    m.match_id,
                    m.match_date,
                    m.competition_name,
                    m.season_name,
                    m.venue,
                    m.home_team_id,
                    m.away_team_id,
                    COALESCE(m.home_team_name, ht.team_name) AS home_team_name,
                    COALESCE(m.away_team_name, at.team_name) AS away_team_name,
                    m.match_label,
                    ht.short_name AS home_short_name,
                    at.short_name AS away_short_name
                FROM matches m
                LEFT JOIN teams ht ON ht.team_id = m.home_team_id
                LEFT JOIN teams at ON at.team_id = m.away_team_id
                ORDER BY m.match_date IS NULL, m.match_date, m.match_id
                LIMIT ? OFFSET ?
                """,
                (page_size, offset),
            ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Database error") from exc

    items: List[Dict[str, Any]] = []
    for row in rows:
        home_name = row_value(row, "home_team_name")
        away_name = row_value(row, "away_team_name")
        kickoff_utc = _normalise_kickoff(row_value(row, "match_date"))
        label = _build_match_label(home_name, away_name, kickoff_utc, row_value(row, "match_label"))

        items.append(
            {
                "id": str(row_value(row, "match_id")) if row_value(row, "match_id") is not None else None,
                "competition": row_value(row, "competition_name"),
                "season": row_value(row, "season_name"),
                "kickoff_utc": kickoff_utc,
                "home_team": team_payload(row_value(row, "home_team_id"), home_name, row_value(row, "home_short_name")),
                "away_team": team_payload(row_value(row, "away_team_id"), away_name, row_value(row, "away_short_name")),
                "venue": row_value(row, "venue"),
                "label": label,
            }
        )

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
    }


__all__ = ["list_matches"]
