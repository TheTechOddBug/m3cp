from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

from fastmcp import FastMCP
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate
from pydantic import ValidationError

from .config import Settings, load_settings
from .errors import (
    INTERNAL_ERROR,
    INVALID_ARGUMENT,
    OPENAI_ERROR,
    UNSUPPORTED_FORMAT,
    MCPError,
    mcp_error,
)
from .fileref import InputData, read_input, write_output_bytes, write_output_text
from .logging_utils import setup_logging
from .openai_client import (
    ImageAnalysisResult,
    ImageGenerationResult,
    OpenAIClient,
    SpeechResult,
    TranscriptionResult,
)
from .schemas import (
    AudioTranscribeArgs,
    AudioTtsArgs,
    ErrorInfo,
    ImageAnalyzeArgs,
    ImageGenerateArgs,
    OutputInfo,
    ToolResult,
)

IMAGE_INPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_INPUT_MIME = {"image/png", "image/jpeg", "image/webp"}
AUDIO_INPUT_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".opus"}
AUDIO_INPUT_MIME = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/opus",
}
IMAGE_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
TTS_OUTPUT_FORMATS = {"mp3", "wav", "opus"}
IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024"}
IMAGE_QUALITY = {"standard", "high"}
IMAGE_BACKGROUND = {"transparent", "opaque"}


class OpenAIClientProtocol(Protocol):
    def analyze_image(
        self,
        image_bytes: bytes,
        instruction: str,
        model_override: Optional[str],
        response_format: str,
        json_schema: Optional[Dict[str, Any]],
        max_output_tokens: Optional[int],
        detail: Optional[str],
        language: Optional[str],
    ) -> ImageAnalysisResult:
        ...

    def generate_image(
        self,
        prompt: str,
        model_override: Optional[str],
        size: Optional[str],
        background: Optional[str],
        quality: Optional[str],
        output_format: Optional[str],
    ) -> ImageGenerationResult:
        ...

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        model_override: Optional[str],
        language: Optional[str],
        prompt: Optional[str],
        timestamps: bool,
        source_filename: Optional[str] = None,
    ) -> TranscriptionResult:
        ...

    def text_to_speech(
        self,
        text: str,
        model_override: Optional[str],
        voice: Optional[str],
        format: Optional[str],
        speed: Optional[float],
    ) -> SpeechResult:
        ...


class ToolService:
    def __init__(
        self,
        settings: Settings,
        client: OpenAIClientProtocol,
        logger: logging.Logger,
    ) -> None:
        self._settings = settings
        self._client = client
        self._logger = logger

    def image_generate(
        self,
        prompt: str,
        output_ref: str,
        size: Optional[str] = None,
        background: Optional[str] = None,
        quality: Optional[str] = None,
        format: Optional[str] = None,
        overwrite: bool = False,
        seed: Optional[int] = None,
        safety: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate an image from a prompt and write it to the output reference."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("image_generate start", request_id, tool="image_generate")
        try:
            args = ImageGenerateArgs(
                prompt=prompt,
                output_ref=output_ref,
                size=size,
                background=background,
                quality=quality,
                format=format,
                overwrite=overwrite,
                seed=seed,
                safety=safety,
                model=model,
                output_headers=output_headers,
            )
            if not args.prompt.strip():
                raise mcp_error(INVALID_ARGUMENT, "prompt is required")
            if args.size and args.size not in IMAGE_SIZES:
                raise mcp_error(INVALID_ARGUMENT, f"Unsupported size: {args.size}")
            if args.background and args.background not in IMAGE_BACKGROUND:
                raise mcp_error(INVALID_ARGUMENT, f"Unsupported background: {args.background}")
            if args.quality and args.quality not in IMAGE_QUALITY:
                raise mcp_error(INVALID_ARGUMENT, f"Unsupported quality: {args.quality}")
            output_format = _normalize_image_format(args.format)
            if args.seed is not None:
                warnings.append("seed is not supported and was ignored")
            if args.safety:
                warnings.append("safety parameters are not supported and were ignored")
            result = self._client.generate_image(
                prompt=args.prompt,
                model_override=args.model,
                size=args.size,
                background=args.background,
                quality=args.quality,
                output_format=output_format,
            )
            mime_type = _image_mime_type(output_format)
            _kind, bytes_written, sha256, path_or_url = write_output_bytes(
                ref=args.output_ref,
                data=result.data,
                mime_type=mime_type,
                settings=self._settings,
                overwrite=args.overwrite,
                headers=args.output_headers,
            )
            output_info = OutputInfo(
                kind="image",
                path_or_url=path_or_url,
                mime_type=mime_type,
                bytes_written=bytes_written,
                sha256=sha256,
            )
            metadata = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_image,
                "duration_ms": result.duration_ms,
                "size": args.size,
                "format": output_format or "png",
            }
            return ToolResult(
                ok=True,
                outputs=[output_info],
                metadata=metadata,
                warnings=warnings,
            ).model_dump()
        except ValidationError as exc:
            return self._error_result(request_id, exc, warnings)
        except MCPError as exc:
            return self._error_result(request_id, exc, warnings)
        except Exception as exc:
            return self._error_result(request_id, exc, warnings)
        finally:
            self._log_info("image_generate end", request_id, tool="image_generate")

    def image_analyze(
        self,
        image_ref: str,
        instruction: str,
        response_format: str = "text",
        json_schema: Optional[Dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        detail: Optional[str] = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze an image and return text or schema-validated JSON."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("image_analyze start", request_id, tool="image_analyze")
        try:
            args = ImageAnalyzeArgs(
                image_ref=image_ref,
                instruction=instruction,
                response_format=response_format,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
                detail=detail,
                language=language,
                model=model,
            )
            if args.response_format not in {"text", "json"}:
                raise mcp_error(INVALID_ARGUMENT, "response_format must be text or json")
            if args.response_format == "json" and not args.json_schema:
                raise mcp_error(INVALID_ARGUMENT, "json_schema is required for json responses")
            if args.detail and args.detail not in {"low", "high"}:
                raise mcp_error(INVALID_ARGUMENT, f"Unsupported detail: {args.detail}")
            if not args.instruction.strip():
                raise mcp_error(INVALID_ARGUMENT, "instruction is required")
            input_data = read_input(args.image_ref, self._settings)
            _validate_image_input(args.image_ref, input_data)
            result = self._client.analyze_image(
                image_bytes=input_data.data,
                instruction=args.instruction,
                model_override=args.model,
                response_format=args.response_format,
                json_schema=args.json_schema,
                max_output_tokens=args.max_output_tokens,
                detail=args.detail,
                language=args.language,
            )
            metadata: Dict[str, Any] = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_vision,
                "duration_ms": result.duration_ms,
                "input_bytes": input_data.size,
            }
            if args.response_format == "json":
                json_schema = args.json_schema
                if json_schema is None:
                    raise mcp_error(INVALID_ARGUMENT, "json_schema is required for json responses")
                if result.json_data is None:
                    raise mcp_error(OPENAI_ERROR, "Expected JSON response from model")
                try:
                    jsonschema_validate(result.json_data, json_schema)
                except JsonSchemaValidationError as exc:
                    raise mcp_error(OPENAI_ERROR, "Model output failed JSON schema validation", exc)
                metadata["json"] = result.json_data
            else:
                metadata["text"] = result.text
            return ToolResult(
                ok=True,
                outputs=[],
                metadata=metadata,
                warnings=warnings,
            ).model_dump()
        except ValidationError as exc:
            return self._error_result(request_id, exc, warnings)
        except MCPError as exc:
            return self._error_result(request_id, exc, warnings)
        except Exception as exc:
            return self._error_result(request_id, exc, warnings)
        finally:
            self._log_info("image_analyze end", request_id, tool="image_analyze")

    def audio_transcribe(
        self,
        audio_ref: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        timestamps: bool = False,
        diarize: bool = False,
        output_ref: Optional[str] = None,
        overwrite: bool = False,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio to text and optionally write the transcript to a file."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("audio_transcribe start", request_id, tool="audio_transcribe")
        try:
            args = AudioTranscribeArgs(
                audio_ref=audio_ref,
                language=language,
                prompt=prompt,
                timestamps=timestamps,
                diarize=diarize,
                output_ref=output_ref,
                overwrite=overwrite,
                model=model,
                output_headers=output_headers,
            )
            input_data = read_input(args.audio_ref, self._settings)
            _validate_audio_input(args.audio_ref, input_data)
            # Extract filename from source for OpenAI format detection
            from pathlib import Path
            source_filename = Path(input_data.source).name if input_data.source else None
            result = self._client.transcribe_audio(
                audio_bytes=input_data.data,
                model_override=args.model,
                language=args.language,
                prompt=args.prompt,
                timestamps=args.timestamps,
                source_filename=source_filename,
            )
            if args.timestamps and not result.segments:
                warnings.append("timestamps requested but not supported by selected model")
            if args.diarize:
                warnings.append("diarize requested but not supported by selected model")
            metadata: Dict[str, Any] = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_stt,
                "duration_ms": result.duration_ms,
                "input_bytes": input_data.size,
            }
            outputs: List[OutputInfo] = []
            if args.output_ref:
                _kind, bytes_written, sha256, path_or_url = write_output_text(
                    ref=args.output_ref,
                    text=result.text,
                    settings=self._settings,
                    overwrite=args.overwrite,
                    headers=args.output_headers,
                )
                output_info = OutputInfo(
                    kind="transcript",
                    path_or_url=path_or_url,
                    mime_type="text/plain; charset=utf-8",
                    bytes_written=bytes_written,
                    sha256=sha256,
                )
                outputs.append(output_info)
            else:
                metadata["text"] = result.text
            if result.segments is not None:
                metadata["segments"] = result.segments
            return ToolResult(
                ok=True,
                outputs=outputs,
                metadata=metadata,
                warnings=warnings,
            ).model_dump()
        except ValidationError as exc:
            return self._error_result(request_id, exc, warnings)
        except MCPError as exc:
            return self._error_result(request_id, exc, warnings)
        except Exception as exc:
            return self._error_result(request_id, exc, warnings)
        finally:
            self._log_info("audio_transcribe end", request_id, tool="audio_transcribe")

    def audio_tts(
        self,
        text: str,
        output_ref: str,
        voice: Optional[str] = None,
        format: Optional[str] = None,
        speed: Optional[float] = None,
        overwrite: bool = False,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate speech audio from text and write it to the output reference."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("audio_tts start", request_id, tool="audio_tts")
        try:
            args = AudioTtsArgs(
                text=text,
                output_ref=output_ref,
                voice=voice,
                format=format,
                speed=speed,
                overwrite=overwrite,
                model=model,
                output_headers=output_headers,
            )
            if not args.text.strip():
                raise mcp_error(INVALID_ARGUMENT, "text is required")
            output_format = _normalize_tts_format(args.format)
            speed_value, speed_warning = _clamp_speed(args.speed)
            if speed_warning:
                warnings.append(speed_warning)
            result = self._client.text_to_speech(
                text=args.text,
                model_override=args.model,
                voice=args.voice,
                format=output_format,
                speed=speed_value,
            )
            mime_type = _tts_mime_type(output_format)
            _kind, bytes_written, sha256, path_or_url = write_output_bytes(
                ref=args.output_ref,
                data=result.data,
                mime_type=mime_type,
                settings=self._settings,
                overwrite=args.overwrite,
                headers=args.output_headers,
            )
            output_info = OutputInfo(
                kind="audio",
                path_or_url=path_or_url,
                mime_type=mime_type,
                bytes_written=bytes_written,
                sha256=sha256,
            )
            metadata = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_tts,
                "duration_ms": result.duration_ms,
                "format": output_format or "mp3",
                "voice": args.voice,
                "speed": speed_value,
            }
            return ToolResult(
                ok=True,
                outputs=[output_info],
                metadata=metadata,
                warnings=warnings,
            ).model_dump()
        except ValidationError as exc:
            return self._error_result(request_id, exc, warnings)
        except MCPError as exc:
            return self._error_result(request_id, exc, warnings)
        except Exception as exc:
            return self._error_result(request_id, exc, warnings)
        finally:
            self._log_info("audio_tts end", request_id, tool="audio_tts")

    def _new_request_id(self) -> str:
        return str(uuid.uuid4())

    def _log_info(self, message: str, request_id: str, **fields: Any) -> None:
        self._logger.info(message, extra={"request_id": request_id, **fields})

    def _log_error(self, message: str, request_id: str, **fields: Any) -> None:
        self._logger.error(message, extra={"request_id": request_id, **fields})

    def _error_result(
        self,
        request_id: str,
        exc: Exception,
        warnings: List[str],
    ) -> Dict[str, Any]:
        if isinstance(exc, ValidationError):
            error = ErrorInfo(code=INVALID_ARGUMENT, message="Invalid arguments")
            self._log_error("validation error", request_id, detail=str(exc))
        elif isinstance(exc, MCPError):
            error = ErrorInfo(code=exc.code, message=exc.message)
            self._log_error("mcp error", request_id, code=exc.code, detail=str(exc))
        else:
            error = ErrorInfo(code=INTERNAL_ERROR, message="Internal error")
            self._logger.error(
                "unhandled error",
                extra={
                    "request_id": request_id,
                    "exception_type": type(exc).__name__,
                    "detail": str(exc),
                },
                exc_info=True
            )
        return ToolResult(
            ok=False,
            outputs=[],
            metadata={"request_id": request_id},
            warnings=warnings,
            error=error,
        ).model_dump()


def build_server(
    settings: Optional[Settings] = None,
    client: Optional[OpenAIClientProtocol] = None,
    logger: Optional[logging.Logger] = None,
) -> FastMCP:
    settings = settings or load_settings()
    logger = logger or setup_logging(settings.log_level)
    client = client or OpenAIClient(settings)
    service = ToolService(settings, client, logger)
    mcp = FastMCP("multimodal-mcp")
    mcp.tool(
        description="Generate an image from a prompt and write it to the output reference.",
    )(service.image_generate)
    mcp.tool(
        description="Analyze an image and return text or schema-validated JSON.",
    )(service.image_analyze)
    mcp.tool(
        description="Transcribe audio to text and optionally write the transcript to a file.",
    )(service.audio_transcribe)
    mcp.tool(
        description="Generate speech audio from text and write it to the output reference.",
    )(service.audio_tts)
    return mcp


def _normalize_image_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "jpg":
        lowered = "jpeg"
    if lowered not in IMAGE_OUTPUT_FORMATS:
        raise mcp_error(UNSUPPORTED_FORMAT, f"Unsupported image format: {value}")
    return lowered


def _normalize_tts_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered not in TTS_OUTPUT_FORMATS:
        raise mcp_error(UNSUPPORTED_FORMAT, f"Unsupported audio format: {value}")
    return lowered


def _image_mime_type(format: Optional[str]) -> str:
    if format == "webp":
        return "image/webp"
    if format == "jpeg":
        return "image/jpeg"
    return "image/png"


def _tts_mime_type(format: Optional[str]) -> str:
    if format == "wav":
        return "audio/wav"
    if format == "opus":
        return "audio/opus"
    return "audio/mpeg"


def _clamp_speed(value: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    if value is None:
        return None, None
    min_speed = 0.25
    max_speed = 4.0
    if value < min_speed:
        return min_speed, "speed clamped to minimum 0.25"
    if value > max_speed:
        return max_speed, "speed clamped to maximum 4.0"
    return value, None


def _validate_image_input(ref: str, input_data: InputData) -> None:
    extension = _extension_from_ref(ref)
    if extension:
        if extension not in IMAGE_INPUT_EXTS:
            raise mcp_error(UNSUPPORTED_FORMAT, f"Unsupported image input: {extension}")
        return
    if input_data.mime_type not in IMAGE_INPUT_MIME:
        raise mcp_error(UNSUPPORTED_FORMAT, "Unsupported image input type")


def _validate_audio_input(ref: str, input_data: InputData) -> None:
    extension = _extension_from_ref(ref)
    if extension:
        if extension not in AUDIO_INPUT_EXTS:
            raise mcp_error(UNSUPPORTED_FORMAT, f"Unsupported audio input: {extension}")
        return
    if input_data.mime_type not in AUDIO_INPUT_MIME:
        raise mcp_error(UNSUPPORTED_FORMAT, "Unsupported audio input type")


def _extension_from_ref(ref: str) -> Optional[str]:
    parsed = urlparse(ref)
    path = parsed.path if parsed.scheme else ref
    extension = Path(path).suffix.lower()
    return extension or None
