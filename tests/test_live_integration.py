from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import pytest

from multimodal_mcp.config import load_settings
from multimodal_mcp.openai_client import OpenAIClient
from multimodal_mcp.server import ToolService


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "gLbq3gAAAABJRU5ErkJggg=="
)


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Live tests disabled")
def test_live_image_analyze(tmp_path: Path) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    if not settings.openai_model_vision:
        pytest.skip("OPENAI_MODEL_VISION is not set")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(PNG_1X1)
    client = OpenAIClient(settings)
    service = ToolService(settings, client, logging.getLogger("live"))
    result = service.image_analyze(
        image_ref=str(image_path),
        instruction="Describe the image",
        response_format="text",
    )
    assert result["ok"] is True
    assert "text" in result["metadata"]
