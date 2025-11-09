"""Utilities for launching and monitoring the local vLLM server."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class VLLMServerConfig:
    model_name: str
    host: str
    port: int
    timeout_seconds: int = 900
    quantization: str | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None


class VLLMServer:
    """Manage a vLLM server process lifecycle."""

    def __init__(self, config: VLLMServerConfig) -> None:
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def start(self) -> None:
        if self._process is not None:
            return
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model_name,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]
        if self.config.quantization:
            command.extend(["--quantization", self.config.quantization])
        if self.config.gpu_memory_utilization is not None:
            command.extend(
                [
                    "--gpu-memory-utilization",
                    str(self.config.gpu_memory_utilization),
                ]
            )
        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])
        env = os.environ.copy()
        env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
        self._process = subprocess.Popen(command, env=env)
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
        finally:
            self._process = None

    def _wait_until_ready(self) -> None:
        start = time.time()
        timeout = self.config.timeout_seconds
        with httpx.Client() as client:
            while True:
                if self._process and self._process.poll() is not None:
                    raise RuntimeError("vLLM server exited unexpectedly while starting")
                try:
                    response = client.get(f"{self.base_url}/health", timeout=10)
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
                if time.time() - start > timeout:
                    raise TimeoutError("Timed out waiting for vLLM server to become ready")
                time.sleep(2)


__all__ = ["VLLMServer", "VLLMServerConfig"]
