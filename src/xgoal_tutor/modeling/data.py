"""Utilities for loading StatsBomb data used in xGoal modeling."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

WIDE_CTE_SQL = """
WITH ff AS (
    SELECT
        shot_id,
        COUNT(*)                                                AS ff_count,
        SUM(CASE WHEN teammate = 1 THEN 1 ELSE 0 END)          AS ff_teammates,
        SUM(CASE WHEN teammate = 0 THEN 1 ELSE 0 END)          AS ff_opponents,
        SUM(CASE WHEN keeper   = 1 THEN 1 ELSE 0 END)          AS ff_keeper_count,
        AVG(CASE WHEN keeper   = 1 THEN x END)                 AS ff_keeper_x,
        AVG(CASE WHEN keeper   = 1 THEN y END)                 AS ff_keeper_y
    FROM freeze_frames
    GROUP BY shot_id
)
SELECT
    s.*,
    e.under_pressure       AS event_under_pressure,
    e.counterpress         AS event_counterpress,
    e.duration             AS event_duration,
    ff.ff_count,
    ff.ff_teammates,
    ff.ff_opponents,
    ff.ff_keeper_count,
    ff.ff_keeper_x,
    ff.ff_keeper_y,
    kp.under_pressure      AS pass_under_pressure,
    json_extract(kp.raw_json, '$.pass.height.name')      AS pass_height,
    json_extract(kp.raw_json, '$.pass.cross')            AS pass_is_cross,
    json_extract(kp.raw_json, '$.pass.through_ball')     AS pass_is_through_ball,
    json_extract(kp.raw_json, '$.pass.cut_back')         AS pass_is_cutback,
    json_extract(kp.raw_json, '$.pass.switch')           AS pass_is_switch
FROM shots s
LEFT JOIN events e  ON e.event_id  = s.shot_id
LEFT JOIN events kp ON kp.event_id = s.key_pass_id
LEFT JOIN ff        ON ff.shot_id  = s.shot_id;
"""


def load_wide_df(db_path: Path | str, use_materialized: bool = True) -> pd.DataFrame | None:
    """Load the wide shot table from a StatsBomb SQLite export."""

    db_path = Path(db_path)
    if not db_path.exists():
        logger.info("DB not found at %s. Proceeding with skeleton only.", db_path.resolve())
        return None

    with sqlite3.connect(db_path) as conn:
        tables = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"])
        if "shots_wide" in tables and use_materialized:
            df = pd.read_sql("SELECT * FROM shots_wide;", conn)
        else:
            df = pd.read_sql(WIDE_CTE_SQL, conn)

    if not df.empty and not df["shot_id"].is_unique:
        raise ValueError("shot_id must be unique (one row per shot)")

    return df
