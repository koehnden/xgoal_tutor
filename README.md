# xG Tutor

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. To
set up a local development environment, install Poetry and then run:

```bash
poetry install
```

Poetry will create (or reuse) a virtual environment and install all required
dependencies from `pyproject.toml`.

## Ingestion pipeline

You can ingest one or more StatsBomb open-data event files—either from your
local machine or directly from the public GitHub repository—using the
`ingest_statsbomb.py` helper script:

```bash
poetry run python src/scripts/ingest_statsbomb.py \
  data/events/123456.json \
  --database xgoal-db.sqlite
```

The script accepts:

- A path to a single local JSON events file.
- A directory containing multiple JSON files (it will recurse through the
  directory and load each file).
- A GitHub tree URL such as
  `https://github.com/statsbomb/open-data/tree/master/data/events`, in which case
  all JSON files within that directory are downloaded temporarily and ingested.

Use the `--database` option to specify the SQLite file to populate (it defaults
to `xgoal-db.sqlite`). Add `--stop-on-error` to halt immediately when an import
fails instead of continuing with the remaining files.
