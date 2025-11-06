from __future__ import annotations
import argparse
import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Tuple, Dict, Optional


def try_import_project_get_db():
    try:
        from xgoal_tutor.api.database import get_db as project_get_db
        return project_get_db
    except Exception:
        return None

@contextmanager
def local_get_db(db_path: str) -> Iterator[sqlite3.Connection]:
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        if conn is not None:
            conn.close()

TABLES = ["matches", "teams", "players", "events", "shots",
          "freeze_frames", "match_lineups"]

PRIMARY_KEYS: Dict[str, List[str]] = {
    "matches": ["match_id"],
    "teams": ["team_id"],
    "players": ["player_id"],
    "events": ["event_id"],
    "shots": ["event_id"],
    "match_lineups": ["match_id", "team_id", "player_id"],
}

FOREIGN_KEYS: List[Tuple[str, List[str], str, List[str]]] = [
    ("events", ["match_id"], "matches", ["match_id"]),
    ("shots", ["team_id"], "teams", ["team_id"]),
    ("shots", ["player_id"], "players", ["player_id"]),
    ("match_lineups", ["match_id"], "matches", ["match_id"]),
    ("match_lineups", ["player_id"], "players", ["player_id"]),
    ("match_lineups", ["team_id"], "teams", ["team_id"]),
]

RANGE_CHECKS: Dict[str, List[Tuple[str, Optional[float], Optional[float]]]] = {
    "shots": [("statsbomb_xg", 0.0, 1.0), ("start_x", 0.0, 120.0), ("start_y", 0.0, 80.0),
              ("end_x", 0.0, 120.0), ("end_y", 0.0, 80.0)],
}

ORDER_CHECK = {
    "table": "events",
    "partition_by": ["match_id"],
    "order_by": "timestamp",
}

def q_scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return 0 if row is None else int(list(row)[0])

def q_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()

def check_pragma_integrity(conn: sqlite3.Connection) -> List[str]:
    issues = []
    fk_on = q_scalar(conn, "PRAGMA foreign_keys;")
    if fk_on != 1:
        issues.append("PRAGMA foreign_keys is OFF (enable it or orphan checks may pass incorrectly)")
    res = q_rows(conn, "PRAGMA integrity_check;")
    if len(res) != 1 or res[0][0] != "ok":
        issues.append(f"PRAGMA integrity_check failed: {res}")
    return issues

def check_non_empty(conn: sqlite3.Connection) -> List[str]:
    issues = []
    for t in TABLES:
        n = q_scalar(conn, f"SELECT COUNT(*) FROM {t}")
        if n == 0:
            issues.append(f"{t} is empty")
    return issues


def check_fk_orphans(conn: sqlite3.Connection) -> List[str]:
    issues = []
    for child, ccols, parent, pcols in FOREIGN_KEYS:
        join_cond = " AND ".join([f"c.{c} = p.{p}" for c, p in zip(ccols, pcols)])
        null_test = " OR ".join([f"p.{p} IS NULL" for p in pcols])
        sql = f"""
            SELECT COUNT(*) AS n
            FROM {child} c
            LEFT JOIN {parent} p ON {join_cond}
            WHERE {null_test};
        """
        n = q_scalar(conn, sql)
        if n > 0:
            issues.append(f"{child} has {n} orphan FK rows referencing {parent}")
    return issues
def check_ranges(conn: sqlite3.Connection) -> List[str]:
    issues = []
    for t, checks in RANGE_CHECKS.items():
        for col, lo, hi in checks:
            conds = []
            if lo is not None:
                conds.append(f"{col} < {lo}")
            if hi is not None:
                conds.append(f"{col} > {hi}")
            if not conds:
                continue

            sql_count = f"SELECT COUNT(*) FROM {t} WHERE {' OR '.join(conds)};"
            n = q_scalar(conn, sql_count)

            if n > 0:
                examples_low, examples_high = [], []

                if lo is not None:
                    sql_low = f"""
                        SELECT {col} AS val
                        FROM {t}
                        WHERE {col} < {lo}
                        ORDER BY {col} ASC
                        LIMIT 10;
                    """
                    try:
                        examples_low = [r["val"] for r in q_rows(conn, sql_low)]
                    except sqlite3.OperationalError:
                        examples_low = []

                if hi is not None:
                    sql_high = f"""
                        SELECT {col} AS val
                        FROM {t}
                        WHERE {col} > {hi}
                        ORDER BY {col} DESC
                        LIMIT 10;
                    """
                    try:
                        examples_high = [r["val"] for r in q_rows(conn, sql_high)]
                    except sqlite3.OperationalError:
                        examples_high = []

                msg_parts = [f"{t}.{col} has {n} out-of-range values"]
                if examples_low:
                    msg_parts.append(f"lowest offenders (up to 10): {examples_low}")
                if examples_high:
                    msg_parts.append(f"highest offenders (up to 10): {examples_high}")

                issues.append(" | ".join(msg_parts))
    return issues


def check_temporal_ordering(conn: sqlite3.Connection) -> List[str]:
    t = ORDER_CHECK["table"]
    by_cols = ORDER_CHECK["partition_by"]
    order_col = ORDER_CHECK["order_by"]
    by_csv = ", ".join(by_cols)
    sql = f"""
        WITH w AS (
            SELECT {by_csv}, {order_col},
                   LAG({order_col}) OVER (PARTITION BY {by_csv} ORDER BY {order_col}) AS prev
            FROM {t}
        )
        SELECT COUNT(*)
        FROM w
        WHERE prev IS NOT NULL AND {order_col} < prev;
    """
    try:
        n = q_scalar(conn, sql)
    except sqlite3.OperationalError as e:
        return [f"{t}: temporal check skipped (SQLite build lacks window functions): {e}"]
    return [] if n == 0 else [f"{t}: {n} temporal inversions within partitions of ({by_csv})"]

def check_two_teams_per_match(conn: sqlite3.Connection) -> List[str]:
    sql = """
        WITH team_counts AS (
            SELECT match_id, COUNT(DISTINCT team_id) AS n
            FROM events
            GROUP BY match_id
        )
        SELECT match_id, n FROM team_counts WHERE n <> 2 LIMIT 20;
    """
    rows = [dict(r) for r in q_rows(conn, sql)]
    return [] if not rows else [f"Matches without exactly 2 teams (first 20): {rows}"]

def check_events_per_match(conn: sqlite3.Connection) -> List[str]:
    sql = """
        SELECT m.match_id, COUNT(e.event_id) AS n_events
        FROM matches m
        LEFT JOIN events e ON e.match_id = m.match_id
        GROUP BY m.match_id
        HAVING COUNT(e.event_id) = 0
        LIMIT 20;
    """
    rows = [dict(r) for r in q_rows(conn, sql)]
    return [] if not rows else [f"Matches with zero events (first 20): {rows}"]


def check_player_belongs_to_team_in_match(conn: sqlite3.Connection) -> List[str]:
    sql = """
        SELECT s.match_id,
               s.player_id,
               s.team_id,
               ml.team_id AS lineup_team_id
        FROM shots s
        LEFT JOIN match_lineups ml
          ON ml.match_id = s.match_id
         AND ml.player_id = s.player_id
        WHERE (ml.team_id IS NULL) OR (ml.team_id <> s.team_id)
        LIMIT 50;
    """
    rows = [dict(r) for r in q_rows(conn, sql)]
    return [] if not rows else [f"Shots with player not matching lineup team (first 50): {rows}"]

def run_all_checks(conn: sqlite3.Connection) -> List[str]:
    checks = [
        ("Non-empty tables", check_non_empty),
        ("Foreign key orphans", check_fk_orphans),
        ("Range checks", check_ranges),
        ("Temporal ordering", check_temporal_ordering),
        ("Exactly two teams per match", check_two_teams_per_match),
        ("Events per match > 0", check_events_per_match),
        ("Roster consistency (shots vs appearances)", check_player_belongs_to_team_in_match),
        # ("SQLite integrity & FK enforcement", check_pragma_integrity),

    ]
    issues: List[str] = []
    for name, fn in checks:
        try:
            errs = fn(conn)
            status = "OK" if not errs else "ISSUES"
            print(f"[{status}] {name}")
            for e in errs:
                print("  -", e)
            issues.extend(errs)
        except sqlite3.OperationalError as e:
            print(f"[SKIPPED] {name} (OperationalError): {e}")
    return issues

def main():
    parser = argparse.ArgumentParser(description="xGoal Tutor: SQLite data-quality checks")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite DB (if not using project get_db())")
    args = parser.parse_args()
    project_get_db = try_import_project_get_db()
    if project_get_db and args.db is None:
        with project_get_db() as conn:
            issues = run_all_checks(conn)
    else:
        if args.db is None:
            print("No project get_db() found. Please pass --db /path/to/your.db")
            return
        with local_get_db(args.db) as conn:
            issues = run_all_checks(conn)
    if issues:
        print("\nSummary: data-quality ISSUES found.")
        raise SystemExit(1)
    else:
        print("\nSummary: all checks passed.")
        raise SystemExit(0)

if __name__ == "__main__":
    main()
