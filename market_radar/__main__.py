"""Entrypoint for running the news aggregator application."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn

from .config import load_config
from .model_server import VLLMServer, VLLMServerConfig
from .model_worker import ModelWorker
from .server import create_app
from .service import NewsAggregator


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()

    server_config = VLLMServerConfig(
        model_name=config.model_name,
        host=config.model_host,
        port=config.model_port,
        quantization=config.model_quantization,
    )
    vllm_server = VLLMServer(server_config)
    logging.info("Starting vLLM server for model %s", server_config.model_name)
    vllm_server.start()

    model_worker = ModelWorker(vllm_server.base_url, config.model_name)
    model_worker.start()

    root = Path(__file__).resolve().parent.parent
    sources_path = root / "sources.json"
    aggregator = NewsAggregator(config, model_worker, sources_path)
    aggregator.start()

    app = create_app(aggregator)

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - runtime hook
        aggregator.stop()
        model_worker.stop()
        vllm_server.stop()

    logging.info("Starting API server on %s:%s", config.api_host, config.api_port)
    try:
        uvicorn.run(app, host=config.api_host, port=config.api_port)
    finally:
        aggregator.stop()
        model_worker.stop()
        vllm_server.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
