# xG Tutor

## Overview
xG Tutor is an end-to-end teaching tool that helps analysts explore expected-goal (xG) insights. It ingests StatsBomb open-data event files, trains a logistic-regression model to estimate shot quality, serves predictions through a FastAPI service, and leverages a local large language model (LLM) to produce plain-language explanations alongside structured metrics.

## Architecture and flow
1. **StatsBomb ETL (`src/xgoal_tutor/etl`)** – Utilities download match event JSON from either local paths or the public StatsBomb GitHub repository, normalise the schema, and persist the data for downstream modelling. The `ingest_statsbomb.py` script under `scripts/` orchestrates these helpers for bulk loading.
2. **Feature engineering and modelling (`src/xgoal_tutor/modeling`)** – Feature builders transform raw events into model-ready tensors. The logistic-regression model applies calibrated coefficients to generate xG probabilities and reason codes for the top contributing features.
3. **Explanation pipeline (`src/xgoal_tutor/llm`)** – The LLM layer wraps a locally running [Ollama](https://ollama.com/) server. Prediction outputs are converted into templated prompts so the chosen Ollama model can produce analyst-friendly summaries. Fallback models ensure robustness when the preferred model is unavailable.
4. **Inference API (`src/xgoal_tutor/api`)** – A FastAPI application exposes prediction endpoints. Incoming shot payloads are scored by the model, cached by match, and optionally enriched with custom prompts before being returned to clients.

The typical flow is: ingest StatsBomb events → engineer features → score shots via the logistic-regression model → request human-readable context from the LLM → retrieve results through the FastAPI endpoints.

## API surface
The FastAPI app exposes three primary routes:
- `POST /predict_shots` – Score one or more shots and automatically request an explanation from the LLM.
- `POST /predict_shots_with_prompt` – Same as above, but accepts a custom prompt override.
- `GET /match/{match_id}/shots` – Retrieve the most recent cached predictions for a match.

Refer to [`docs/swagger.yaml`](docs/swagger.yaml) for the OpenAPI definition and [`docs/xgoal_postman_collection.json`](docs/xgoal_postman_collection.json) for ready-made Postman examples covering request payloads and typical responses.

## Running the FastAPI service
1. **Install dependencies**
   ```bash
   poetry install
   ```
2. **Start an Ollama server** – Install Ollama and run `ollama serve` (or ensure an existing instance is available at `http://localhost:11434`). Pull the primary and fallback models referenced in `src/xgoal_tutor/api/models.py` using `ollama pull <model-name>`.
3. **Launch the API**
   ```bash
   poetry run uvicorn xgoal_tutor.api.app:app --reload
   ```
   The service hosts interactive docs at `http://127.0.0.1:8000/docs`.

## Running the ingestion helper
Use the StatsBomb ingestion script to populate a SQLite database with open-data events:

```bash
poetry run python scripts/ingest_statsbomb.py \
  data/events/123456.json \
  --database xgoal-db.sqlite
```

The script accepts individual files, directories, or GitHub tree URLs and supports a `--stop-on-error` flag to halt on the first failure.

## Testing
Execute the unit test suite with:

```bash
poetry run pytest
```

Ensure the Ollama service is reachable when exercising tests that touch the LLM client.
