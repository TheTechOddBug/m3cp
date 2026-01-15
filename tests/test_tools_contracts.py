from __future__ import annotations

import base64
import logging
from pathlib import Path

from multimodal_mcp.config import Settings
from multimodal_mcp.openai_client import (
    AudioAnalysisResult,
    AudioTransformResult,
    ImageAnalysisResult,
    ImageEditResult,
    ImageExtractResult,
    ImageGenerationResult,
    ImageSpecResult,
    SpeechResult,
    TranscriptionResult,
)
from multimodal_mcp.server import ToolService


class FakeOpenAIClient:
    def __init__(
        self,
        image_bytes: bytes = b"img",
        edit_bytes: bytes = b"edit",
        analysis_text: str = "{\"caption\": \"ok\"}",
        analysis_json: dict | None = None,
        extract_json: dict | None = None,
        spec_text: str = "diagram",
        transcript_text: str = "hello",
        audio_analysis_text: str = "analysis",
        audio_analysis_json: dict | None = None,
        tts_bytes: bytes = b"audio",
        transform_bytes: bytes = b"audio2",
    ) -> None:
        self._image_bytes = image_bytes
        self._edit_bytes = edit_bytes
        self._analysis_text = analysis_text
        self._analysis_json = analysis_json
        self._extract_json = extract_json or {"field": "ok"}
        self._spec_text = spec_text
        self._transcript_text = transcript_text
        self._audio_analysis_text = audio_analysis_text
        self._audio_analysis_json = audio_analysis_json
        self._tts_bytes = tts_bytes
        self._transform_bytes = transform_bytes

    def generate_image(self, *args: object, **kwargs: object) -> ImageGenerationResult:
        return ImageGenerationResult(data=self._image_bytes, duration_ms=5)

    def edit_image(self, *args: object, **kwargs: object) -> ImageEditResult:
        return ImageEditResult(data=self._edit_bytes, duration_ms=5)

    def analyze_image(self, *args: object, **kwargs: object) -> ImageAnalysisResult:
        return ImageAnalysisResult(
            text=self._analysis_text,
            json_data=self._analysis_json,
            duration_ms=6,
        )

    def extract_image(self, *args: object, **kwargs: object) -> ImageExtractResult:
        return ImageExtractResult(json_data=self._extract_json, duration_ms=6)

    def image_to_spec(self, *args: object, **kwargs: object) -> ImageSpecResult:
        return ImageSpecResult(text=self._spec_text, duration_ms=6)

    def transcribe_audio(self, *args: object, **kwargs: object) -> TranscriptionResult:
        return TranscriptionResult(text=self._transcript_text, segments=None, duration_ms=7)

    def analyze_audio(self, *args: object, **kwargs: object) -> AudioAnalysisResult:
        return AudioAnalysisResult(
            text=self._audio_analysis_text,
            json_data=self._audio_analysis_json,
            duration_ms=7,
        )

    def transform_audio(self, *args: object, **kwargs: object) -> AudioTransformResult:
        return AudioTransformResult(data=self._transform_bytes, duration_ms=7)

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
        openai_model_image_edit="image-edit",
        openai_model_stt="stt",
        openai_model_tts="tts",
        openai_model_audio_analyze="audio-analyze",
        openai_model_audio_transform="audio-transform",
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


def valid_png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )


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


def test_image_extract_schema_valid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(extract_json={"field": "ok"})
    service = ToolService(settings, client, make_logger())
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = service.image_extract(
        image_ref=str(image_path),
        instruction="Extract field",
        json_schema=json_schema,
    )
    assert result["ok"] is True
    assert result["metadata"]["json"]["field"] == "ok"


def test_image_extract_schema_invalid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
        "required": ["field"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(extract_json={"wrong": "nope"})
    service = ToolService(settings, client, make_logger())
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    result = service.image_extract(
        image_ref=str(image_path),
        instruction="Extract field",
        json_schema=json_schema,
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "SCHEMA_VALIDATION_FAILED"


def test_audio_analyze_schema_valid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"sentiment": {"type": "string"}},
        "required": ["sentiment"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(audio_analysis_json={"sentiment": "neutral"})
    service = ToolService(settings, client, make_logger())
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    result = service.audio_analyze(
        audio_ref=str(audio_path),
        instruction="Analyze sentiment",
        response_format="json",
        json_schema=json_schema,
    )
    assert result["ok"] is True
    assert result["metadata"]["json"]["sentiment"] == "neutral"


def test_audio_analyze_schema_invalid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"sentiment": {"type": "string"}},
        "required": ["sentiment"],
        "additionalProperties": False,
    }
    client = FakeOpenAIClient(audio_analysis_json={"wrong": "nope"})
    service = ToolService(settings, client, make_logger())
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    result = service.audio_analyze(
        audio_ref=str(audio_path),
        instruction="Analyze sentiment",
        response_format="json",
        json_schema=json_schema,
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "SCHEMA_VALIDATION_FAILED"


def test_multimodal_chain_resolves_outputs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeOpenAIClient(image_bytes=valid_png_bytes(), edit_bytes=b"edited")
    service = ToolService(settings, client, make_logger())
    output_path = tmp_path / "generated.png"
    edit_output = tmp_path / "edited.png"
    result = service.multimodal_chain(
        steps=[
            {
                "tool": "image_generate",
                "args": {
                    "prompt": "A square",
                    "output_ref": str(output_path),
                    "overwrite": True,
                },
                "outputs_as": "gen",
            },
            {
                "tool": "image_edit",
                "args": {
                    "image_ref": {"$ref": "gen.outputs[0].path_or_url"},
                    "prompt": "Add a border",
                    "output_ref": str(edit_output),
                    "overwrite": True,
                },
                "outputs_as": "edit",
            },
        ],
    )
    assert result["ok"] is True
    assert edit_output.exists()


def test_multimodal_chain_failure_propagates(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeOpenAIClient(image_bytes=valid_png_bytes(), edit_bytes=b"edited")
    service = ToolService(settings, client, make_logger())
    output_path = tmp_path / "generated.png"
    edit_output = tmp_path / "edited.png"
    result = service.multimodal_chain(
        steps=[
            {
                "tool": "image_generate",
                "args": {
                    "prompt": "A square",
                    "output_ref": str(output_path),
                    "overwrite": True,
                },
                "outputs_as": "gen",
            },
            {
                "tool": "image_edit",
                "args": {
                    "image_ref": {"$ref": "gen.outputs[0].path_or_url"},
                    "prompt": "",
                    "output_ref": str(edit_output),
                },
                "outputs_as": "edit",
            },
        ],
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "CHAIN_STEP_FAILED"
