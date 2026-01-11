from __future__ import annotations

import logging
from pathlib import Path

from multimodal_mcp.config import Settings
from multimodal_mcp.openai_client import (
    ImageAnalysisResult,
    ImageGenerationResult,
    SpeechResult,
    TranscriptionResult,
)
from multimodal_mcp.server import ToolService


class FakeOpenAIClient:
    def __init__(
        self,
        image_bytes: bytes = b"img",
        analysis_text: str = "{\"caption\": \"ok\"}",
        analysis_json: dict | None = None,
        transcript_text: str = "hello",
        tts_bytes: bytes = b"audio",
    ) -> None:
        self._image_bytes = image_bytes
        self._analysis_text = analysis_text
        self._analysis_json = analysis_json
        self._transcript_text = transcript_text
        self._tts_bytes = tts_bytes

    def generate_image(self, *args: object, **kwargs: object) -> ImageGenerationResult:
        return ImageGenerationResult(data=self._image_bytes, duration_ms=5)

    def analyze_image(self, *args: object, **kwargs: object) -> ImageAnalysisResult:
        return ImageAnalysisResult(
            text=self._analysis_text,
            json_data=self._analysis_json,
            duration_ms=6,
        )

    def transcribe_audio(self, *args: object, **kwargs: object) -> TranscriptionResult:
        return TranscriptionResult(text=self._transcript_text, segments=None, duration_ms=7)

    def text_to_speech(self, *args: object, **kwargs: object) -> SpeechResult:
        return SpeechResult(data=self._tts_bytes, duration_ms=8)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test",
        openai_base_url=None,
        openai_org_id=None,
        openai_project=None,
        openai_model_vision="vision",
        openai_model_image="image",
        openai_model_stt="stt",
        openai_model_tts="tts",
        enable_remote_urls=False,
        enable_presigned_uploads=False,
        allow_insecure_http=False,
        allow_mkdir=True,
        max_input_bytes=1024,
        max_output_bytes=1024,
        log_level="INFO",
        temp_dir=tmp_path / "temp",
    )


def make_logger() -> logging.Logger:
    logger = logging.getLogger("tests.tools")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def test_image_analyze_json_schema_valid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"caption": {"type": "string"}},
        "required": ["caption"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(analysis_json={"caption": "ok"})
    service = ToolService(settings, client, make_logger())
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = service.image_analyze(
        image_ref=str(image_path),
        instruction="caption",
        response_format="json",
        json_schema=json_schema,
    )
    assert result["ok"] is True
    assert result["metadata"]["json"]["caption"] == "ok"


def test_image_analyze_json_schema_invalid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"caption": {"type": "string"}},
        "required": ["caption"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(analysis_json={"wrong": "nope"})
    service = ToolService(settings, client, make_logger())
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = service.image_analyze(
        image_ref=str(image_path),
        instruction="caption",
        response_format="json",
        json_schema=json_schema,
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "OPENAI_ERROR"


def test_audio_transcribe_writes_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeOpenAIClient(transcript_text="hello")
    service = ToolService(settings, client, make_logger())
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_path = tmp_path / "out.txt"
    result = service.audio_transcribe(
        audio_ref=str(audio_path),
        output_ref=str(output_path),
    )
    assert result["ok"] is True
    assert output_path.read_text(encoding="utf-8") == "hello"
