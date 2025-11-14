# News Aggregator

This project runs a GPU-enabled news aggregation service. It continuously parses
RSS feeds from `sources.json`, generates LLM summaries for the past week's news,
and exposes an HTTP endpoint that returns the most relevant articles for the
topics you supply.

## Features

- **Automated RSS ingestion** – `market_radar.fetching.NewsFetcher` downloads and
  normalises articles from the configured agencies. Items older than one week
  are automatically discarded. 【F:market_radar/fetching.py†L1-L288】
- **LLM-powered summaries** – A dedicated worker thread interacts with a local
  vLLM server hosting one of the Qwen 3 4B instruct checkpoints (FP8 or full
  precision) to produce concise, neutral summaries.
- **Topic-aware reranking** – For every request the service asks the same model
  to score news items against the provided topics and returns the highest scoring
  reports.
- **Single-container deployment** – The Docker image starts both the vLLM server
  and the FastAPI app. Only the API port is exposed; the model server remains
  isolated inside the container.

## Running locally

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export the environment variables you want to tweak (all optional):

   - `PORT` – HTTP port for the FastAPI service (default `8080`).
   - `MODEL_VARIANT` – Shortcut for switching between `fp8` and
     `Qwen/Qwen3-4B-Instruct-2507` full precision weights (`base`). When set it
     overrides `MODEL_NAME`/`MODEL_QUANTIZATION`.
   - `MODEL_NAME` – Hugging Face model id (default `Qwen/Qwen3-4B-Instruct-2507-FP8`).
   - `MODEL_QUANTIZATION` – Optional vLLM quantization argument (default `fp8`).
   - `MODEL_PORT` – internal port for the vLLM server (default `8001`).
   - `MODEL_LOCAL_PATH` – absolute path to a directory containing `config.json`
     and `.safetensors`/`.bin` weights. If unset the service looks for
     `/models` and falls back to remote downloads when missing.
   - `GPU_MEMORY_UTILIZATION` – fraction of each GPU's memory vLLM is allowed
     to allocate (default `0.9`).
   - `MAX_MODEL_LEN` – hard cap for the model context length passed to vLLM
     (default `53712`). Reduce this when running on GPUs with limited memory.
   - `FETCH_INTERVAL_SECONDS` – delay between RSS refresh cycles (default 1800).
   - `SUMMARY_MAX_CHARS` – maximum number of characters sent to the LLM for
     summarisation (default 4000).
   - `VLLM_USE_TORCH_COMPILE` – set to `0` to disable PyTorch compilation if you
     run on a host without a working compiler toolchain. The provided Docker
     image includes the required build utilities so compilation is enabled by
     default.

4. Start the service:

   ```bash
   python -m market_radar
   ```

The process launches the vLLM server, waits for it to finish downloading the
model, starts the summarisation worker, and finally exposes the FastAPI app.

### API reference

The service exposes a small FastAPI application whose schema is described in
[`swagger.yml`](swagger.yml). Below are practical examples that mirror the
behaviour of the deployed API.

#### `POST /news`

Request summarised, reranked news reports for a list of topics. The `topics`
field accepts questions or themes (they are normalised before being sent to the
LLM) and `top_n` limits the amount of news to return.

```bash
curl -X POST http://localhost:8080/news \
  -H "Content-Type: application/json" \
  -d '{
        "topics": ["inflation", "bank earnings"],
        "top_n": 3
      }'
```

Sample response:

```json
[
  {
    "agency": "reuters",
    "title": "Fed signals patience as inflation cools",
    "summary": "Federal Reserve officials flagged a cautious approach to rate cuts after new inflation data showed progress toward the bank's 2% target.",
    "image_base64": null,
    "url": "https://www.reuters.com/markets/us/fed-patience-inflation",
    "published_at": "2024-08-10T12:30:00Z",
    "crawled_at": "2024-08-10T12:31:12Z"
  },
  {
    "agency": "bloomberg",
    "title": "US banks see earnings rebound as provisions shrink",
    "summary": "Major banks reported stronger quarterly profits thanks to lower credit-loss provisions and resilient consumer lending.",
    "image_base64": null,
    "url": "https://www.bloomberg.com/news/articles/bank-earnings-rebound",
    "published_at": "2024-08-09T09:15:00Z",
    "crawled_at": "2024-08-09T09:17:44Z"
  }
]
```

Each entry contains the RSS agency identifier, optional headline, LLM summary,
Base64-encoded lead image, publication timestamp reported by the source, and the
internal crawl timestamp.

#### `POST /reports/rebuild`

Force the service to rebuild its cached summaries from the currently retained
RSS entries. This is useful when you change summarisation limits or want to
reseed the cache after importing historical feeds.

```bash
curl -X POST http://localhost:8080/reports/rebuild
```

Sample response:

```json
{ "rebuilt": 152 }
```

The `rebuilt` value indicates how many reports are currently ready to be served
through `/news`.

#### `POST /advice`

The advice endpoint accepts structured financial inputs, forwards them to the
LLM for additional analysis, and returns a validated payload. It is meant for
turning natural-language wishes into consistent budgeting recommendations.

```bash
curl -X POST http://localhost:8080/advice \
  -H "Content-Type: application/json" \
  -d '{
        "earnings": 5400,
        "wastes": {
          "rent": 2200,
          "groceries": 650,
          "dining": 400,
          "travel": 250
        },
        "wishes": "Reduce dining but leave budget for one weekend getaway"
      }'
```

Sample response:

```json
{
  "earnings": 5400,
  "wastes": {
    "rent": 2200,
    "groceries": 600,
    "dining": 300,
    "travel": 350
  },
  "wishes": "Prioritise savings without cancelling all leisure travel"
}
```

Validation errors return a `400` status code with a JSON body containing a
`detail` message. Internal model failures surface as `502` responses.

## Docker & Compose

A CUDA-enabled Dockerfile is provided alongside two example Compose definitions.
Copy the variant that matches your deployment into `docker-compose.yaml` before
starting the stack.

### Remote model downloads

Use `docker-compose.example.remote.yaml` to let vLLM download the model at
startup:

```bash
cp docker-compose.example.remote.yaml docker-compose.yaml
docker compose up --build
```

### Local model weights

If you already have the Hugging Face repository on disk, mount it into the
container and set `MODEL_LOCAL_PATH` to point at the mount (defaults to
`/models`). The service will validate that the directory contains
`config.json` and tensor weights before using it.

```bash
cp docker-compose.example.local.yaml docker-compose.yaml
# assumes ./models contains the exported Hugging Face repository
docker compose up --build
```

If the directory is missing or does not contain model weights the service falls
back to downloading the model from Hugging Face.

## Architecture overview

- `market_radar/model_server.py` starts and supervises the vLLM process, waiting
  until the OpenAI-compatible HTTP interface reports healthy before continuing.
- `market_radar/model_worker.py` owns a single background thread that serialises
  all CUDA-bound requests. Summaries and reranking prompts are pushed onto a
  queue and executed sequentially to avoid multithreading issues.
- `market_radar/service.py` orchestrates RSS fetching, summary generation, image
  downloads, retention cleanup, and reranking.
- `market_radar/server.py` wires the FastAPI application with the aggregation
  service.

## Development tips

- `python -m compileall market_radar` performs a quick syntax check.
- Environment variables make it easy to shorten the fetch interval or adjust
  summarisation limits when iterating locally.
- Logs include detailed information about fetch cycles and reranking fallbacks.
