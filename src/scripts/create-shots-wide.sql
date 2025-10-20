DROP TABLE IF EXISTS shots_wide;

CREATE TABLE shots_wide AS
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
    ff.ff_keeper_y
FROM shots s
LEFT JOIN events e ON e.event_id = s.shot_id
LEFT JOIN ff     ON ff.shot_id   = s.shot_id
ORDER BY s.match_id, s.minute, s.second, s.shot_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sw_shot_id ON shots_wide(shot_id);
CREATE INDEX IF NOT EXISTS idx_sw_match ON shots_wide(match_id);
