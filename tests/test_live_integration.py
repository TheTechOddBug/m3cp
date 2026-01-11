from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

import pytest

from multimodal_mcp.config import load_settings
from multimodal_mcp.openai_client import OpenAIClient
from multimodal_mcp.server import ToolService


def create_test_image() -> bytes:
    """Create a valid 256x256 test image."""
    try:
        from PIL import Image
        # Create a simple red 256x256 image
        img = Image.new('RGB', (256, 256), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    except ImportError:
        # Fallback to a known valid base64 encoded PNG if PIL is not available
        # This is a valid 256x256 red square
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAAnklEQVR4nO3BMQEAAADCoPVP"
            "bQhfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOD8GDq8AAEkE9CTAAAAASUVORK5CYII="
        )


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Live tests disabled")
def test_live_image_analyze(tmp_path: Path) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    if not settings.openai_model_vision:
        pytest.skip("OPENAI_MODEL_VISION is not set")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(create_test_image())
    client = OpenAIClient(settings)
    service = ToolService(settings, client, logging.getLogger("live"))
    result = service.image_analyze(
        image_ref=str(image_path),
        instruction="Describe the image",
        response_format="text",
    )
    if not result["ok"]:
        print(f"\nTest failed with error: {result.get('error')}")
        print(f"Full result: {result}")
    assert result["ok"] is True, f"API call failed: {result.get('error', 'Unknown error')}"
    assert "text" in result["metadata"]
