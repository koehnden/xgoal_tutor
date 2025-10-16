# xG Tutor

## ETL quickstart

1. Install dependencies with `poetry install`.
2. Download a StatsBomb events JSON file and place it in a working directory.
3. Run the ETL helper to load the match data into a local SQLite database:

   ```bash
   poetry run python -m xgoal_tutor.etl /path/to/events.json /path/to/output.db
   ```

4. Inspect the populated `events`, `shots`, and `freeze_frames` tables using your preferred SQLite browser.

The ETL loader is idempotent and can be executed repeatedly; existing rows are
replaced based on the `event_id` and `shot_id` keys. Freeze-frame rows are
refreshed on each run to ensure the defensive context remains aligned with the
source JSON.
