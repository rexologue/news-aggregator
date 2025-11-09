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

    if config.model_local_path and not config.using_local_model:
        logging.warning(
            "Local model path %s does not look like a valid Hugging Face model directory; "
            "falling back to remote download",
            config.model_local_path,
        )

    model_identifier = config.model_identifier
    if config.using_local_model:
        logging.info("Using local model weights from %s", model_identifier)
    else:
        logging.info("Using Hugging Face model %s", config.model_name)
    logging.info(
        "Model variant: %s (quantization: %s)",
        config.model_variant,
        config.model_quantization or "disabled",
    )

    server_config = VLLMServerConfig(
        model_name=model_identifier,
        host=config.model_host,
        port=config.model_port,
        quantization=config.model_quantization,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    vllm_server = VLLMServer(server_config)
    logging.info("Starting vLLM server for model %s", server_config.model_name)
    vllm_server.start()

    model_worker = ModelWorker(vllm_server.base_url, model_identifier)
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
