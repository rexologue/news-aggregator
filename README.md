# Market Radar

Market Radar is a modular pipeline that collects technology and finance news, scores
articles for topical density, classifies and summarizes them, and produces a
ranked "hotness" view for analysts.

## Key components

- **Fetcher** – downloads articles from configured RSS feeds, normalises
  timestamps, and filters out short or duplicate entries. 【F:market_radar/fetching.py†L1-L288】
- **Density estimator** – embeds titles and article bodies, weights them,
  compares sources within a rolling window, and gives higher scores to pieces
  that are unique relative to peers. 【F:market_radar/density_estimator.py†L1-L200】
- **Summariser** – calls an OpenRouter-compatible chat model (with graceful
  fallback) to generate Russian summaries and topical categories used as domain
  weights. 【F:market_radar/summarizer.py†L1-L129】
- **Hotness calculator** – applies time decay, density and domain weights to
  compute the final ranking, then min-max normalises the results. 【F:market_radar/hotness.py†L1-L59】
- **Orchestrator** – wires all components together, writes structured JSON, and
  can be reused in automation. 【F:market_radar/orchestrator.py†L1-L74】

## Installation

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain an [OpenRouter](https://openrouter.ai/) API key if you want to run the
   LLM summariser and export it as `OPENROUTER_API_KEY`.

## Configuration

The pipeline is configured with a YAML file such as
[`config.example.yaml`](./config.example.yaml). Each section maps to a dataclass
in `market_radar.config` and is validated when loaded. 【F:market_radar/config.py†L12-L138】

- `time_window.since` is parsed with `NewsFetcher.parse_since` and controls how
  far back in time the fetcher looks (e.g. `6h` for six hours). 【F:market_radar/fetching.py†L38-L63】
- `fetcher` covers RSS details, retries, timeouts, and the path to `sources.json`
  that lists feed URLs. 【F:market_radar/fetching.py†L70-L287】
- `density` configures the embedding model, weighting between title and body,
  the amount of body text retained, and the hourly window used for
  cross-article comparisons. 【F:market_radar/density_estimator.py†L43-L199】
- `summarizer` sets the OpenRouter model parameters and controls whether the
  heuristic fallback summary is allowed. 【F:market_radar/summarizer.py†L17-L125】
- `hotness` defines the relative weights for time, density, and domain
  components as well as the exponential time decay factor. 【F:market_radar/hotness.py†L11-L65】
- `output.path` points to the JSON file that the orchestrator writes. 【F:market_radar/orchestrator.py†L53-L76】

Copy the example file and adjust it for your feeds:

```bash
cp config.example.yaml config.yaml
```

## How scoring works

1. **Fetching and time logic** – For every RSS source, the fetcher retries on
   transient errors, deduplicates links, and discards articles published before
   the computed cutoff (`now - since`). Articles retain their best timestamp
   (published date when available or crawl time) which feeds later stages.
   【F:market_radar/fetching.py†L70-L287】【F:market_radar/models.py†L10-L34】
2. **Density computation** – Titles and cleaned article bodies are encoded with
   a Sentence Transformer. Their cosine-normalised embeddings are weighted using
   `title_score` and `content_score`, grouped into rolling windows, and scored by
   measuring how different each article is from others from different sources.
   Higher uniqueness produces higher density coefficients. 【F:market_radar/density_estimator.py†L14-L199】
3. **Summaries and domain weights** – The summariser enforces a strict output
   schema, maps the returned category to predefined weights, and falls back to a
   heuristic summary when the model is unavailable. The category weight becomes
   `domain_coef`. 【F:market_radar/summarizer.py†L21-L125】
4. **Time-aware hotness** – For each article the calculator derives a time
   coefficient using an exponential decay bounded to the configured window. It
   then combines time, density, and domain components with configured weights
   and normalises scores to `[0, 1]`. 【F:market_radar/hotness.py†L13-L65】
5. **Output** – The orchestrator attaches all coefficients, sorts by hotness,
   and writes prettified JSON to the configured location. 【F:market_radar/orchestrator.py†L32-L76】

## Example usage

An executable helper is provided in [`examples/run_pipeline.py`](./examples/run_pipeline.py).
It loads a configuration file, runs the orchestrator, and prints the location of
the generated JSON report. 【F:examples/run_pipeline.py†L1-L38】

```bash
python examples/run_pipeline.py --config config.yaml
```

## HTTP API

The project ships with a FastAPI wrapper that exposes the pipeline over HTTP.
It accepts query parameters to override key configuration values on demand and
returns the ranked articles as JSON.

### Run locally

```bash
python -m market_radar --config $(pwd)/config.yaml --port 8000
```

Send a request to trigger the pipeline:

```bash
curl -X POST \
  "http://localhost:8000/pipeline?&since=6h" \
  --output output.json
```

Available query parameters:

- `--output` – **required**, specifies where the generated JSON report is written.
- `since` – optional override for `time_window.since` (e.g. `6h`).

All other configuration values are sourced exclusively from the YAML file
provided when the service starts.

Set `MARKET_RADAR_MODEL_CACHE` (or the common `HF_HOME`/`TRANSFORMERS_CACHE`)
to reuse a persistent cache for Sentence Transformers weights.

### Container image

The repository includes a [`Dockerfile`](./Dockerfile) and helper script for
building and running the API in a container. To start the service with your
configuration mounted read-only and the Hugging Face cache persisted on the
host, execute:

```bash
PORT=8000 \
CONFIG_PATH=~/market-radar/config.yaml \
MODELS_DIR=~/.cache/huggingface \
./tools/run_api_container.sh
```

Important environment variables:

- `CONFIG_PATH` – path to the YAML config mounted into the container.
- `SOURCES_PATH` – optional path to a custom `sources.json` file.
- `MODELS_DIR` – directory on the host reused as the Hugging Face cache to
  avoid model re-downloads.
- `OPENROUTER_API_KEY` – forwarded to the container when present so the
  summariser can reach OpenRouter.
- `DO_BUILD=1` – build the Docker image before running it.
- `DEV=1` – mount the repository into `/app` for live-editing inside the
  container.

The FastAPI app listens on the port specified via the `PORT` environment
variable (default `8000`).

## Deployment notes

- **Secrets & credentials** – Store the OpenRouter API key and any feed
  credentials in your secret manager and expose them as environment variables at
  runtime. The summariser automatically reads `OPENROUTER_API_KEY`. 【F:market_radar/summarizer.py†L33-L81】
- **Model caching** – Set `density.model_cache_dir` to a writable volume so that
  Sentence Transformer weights persist across runs. 【F:market_radar/density_estimator.py†L43-L73】
- **Scheduling** – Run the example script (or your own wrapper around
  `NewsPipelineOrchestrator`) from cron, Airflow, or another scheduler to keep
  the report up to date.
- **Containerisation** – Package the project into a container, mount the config
  file and sources list, and ensure outbound HTTPS access for RSS feeds and the
  OpenRouter API.
- **Monitoring** – Capture stdout/stderr from the fetcher and summariser to
  detect repeated failures; consider plugging in a logging handler in production.
- **Outputs** – Serve the generated JSON through your API or upload it to a
  storage bucket/CDN as part of your deployment job. The file is overwritten on
  each run, so version it if historical snapshots are required.

## Development tips

- Use `python -m compileall market_radar examples` to perform a quick syntax
  check locally.
- When iterating on the fetcher, reduce `fetcher.max_per_source` and shorten the
  time window for faster cycles.
