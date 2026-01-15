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
        return _create_png_bytes(256, 256, (255, 0, 0))


def _create_png_bytes(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    import struct
    import zlib

    r, g, b = rgb
    row = bytes([0]) + bytes([r, g, b]) * width
    raw = row * height
    compressed = zlib.compress(raw)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return length + chunk_type + data + struct.pack(">I", crc)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            signature,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", compressed),
            _chunk(b"IEND", b""),
        ]
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


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Live tests disabled")
def test_live_image_edit(tmp_path: Path) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    if not settings.openai_model_image_edit:
        pytest.skip("OPENAI_MODEL_IMAGE_EDIT is not set")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(create_test_image())
    output_path = tmp_path / "edited.png"
    client = OpenAIClient(settings)
    service = ToolService(settings, client, logging.getLogger("live"))
    result = service.image_edit(
        image_ref=str(image_path),
        prompt="Add a subtle vignette",
        output_ref=str(output_path),
        overwrite=True,
    )
    if not result["ok"]:
        print(f"\nTest failed with error: {result.get('error')}")
        print(f"Full result: {result}")
    assert result["ok"] is True, f"API call failed: {result.get('error', 'Unknown error')}"
    assert output_path.exists()


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="Live tests disabled")
def test_live_audio_analyze(tmp_path: Path) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    if not settings.openai_model_audio_analyze:
        pytest.skip("OPENAI_MODEL_AUDIO_ANALYZE is not set")
    audio_path = Path(__file__).parent.parent / "mcp-tools.mp3"
    if not audio_path.exists():
        pytest.skip("Sample audio missing")
    client = OpenAIClient(settings)
    service = ToolService(settings, client, logging.getLogger("live"))
    result = service.audio_analyze(
        audio_ref=str(audio_path),
        instruction="Summarize tone and sentiment",
        response_format="text",
    )
    if not result["ok"]:
        print(f"\nTest failed with error: {result.get('error')}")
        print(f"Full result: {result}")
    assert result["ok"] is True, f"API call failed: {result.get('error', 'Unknown error')}"
    assert "text" in result["metadata"]
