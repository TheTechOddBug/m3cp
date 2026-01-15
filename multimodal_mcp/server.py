from __future__ import annotations

import io
import json
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
    CHAIN_STEP_FAILED,
    INTERNAL_ERROR,
    INVALID_ARGUMENT,
    OPENAI_ERROR,
    SCHEMA_VALIDATION_FAILED,
    UNSUPPORTED_FORMAT,
    MCPError,
    mcp_error,
)
from .fileref import InputData, read_input, write_output_bytes, write_output_text
from .logging_utils import setup_logging
from .openai_client import (
    AudioAnalysisResult,
    AudioTransformResult,
    ImageEditResult,
    ImageExtractResult,
    ImageGenerationResult,
    ImageSpecResult,
    OpenAIClient,
    SpeechResult,
    TranscriptionResult,
)
from .schemas import (
    AudioAnalyzeArgs,
    AudioTranscribeArgs,
    AudioTransformArgs,
    AudioTtsArgs,
    ErrorInfo,
    ImageAnalyzeArgs,
    ImageEditArgs,
    ImageExtractArgs,
    ImageGenerateArgs,
    ImageToSpecArgs,
    MultimodalChainArgs,
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
IMAGE_SPEC_FORMATS = {"mermaid", "plantuml", "openapi", "c4", "markdown"}


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

    def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        mask_bytes: Optional[bytes],
        model_override: Optional[str],
        size: Optional[str],
        output_format: Optional[str],
        source_filename: Optional[str],
        mask_filename: Optional[str],
    ) -> ImageEditResult:
        ...

    def extract_image(
        self,
        image_bytes: bytes,
        instruction: str,
        json_schema: Dict[str, Any],
        model_override: Optional[str],
        language: Optional[str],
        max_output_tokens: Optional[int],
    ) -> ImageExtractResult:
        ...

    def image_to_spec(
        self,
        image_bytes: bytes,
        instruction: Optional[str],
        target_format: str,
        model_override: Optional[str],
        max_output_tokens: Optional[int],
    ) -> ImageSpecResult:
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

    def analyze_audio(
        self,
        audio_bytes: bytes,
        instruction: str,
        model_override: Optional[str],
        response_format: str,
        json_schema: Optional[Dict[str, Any]],
        source_filename: Optional[str],
    ) -> AudioAnalysisResult:
        ...

    def transform_audio(
        self,
        audio_bytes: bytes,
        instruction: str,
        model_override: Optional[str],
        voice: Optional[str],
        format: Optional[str],
        source_filename: Optional[str],
    ) -> AudioTransformResult:
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

    def image_edit(
        self,
        image_ref: str,
        prompt: str,
        output_ref: str,
        mask_ref: Optional[str] = None,
        format: Optional[str] = None,
        size: Optional[str] = None,
        overwrite: bool = False,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Edit or inpaint an image and write the result to the output reference."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("image_edit start", request_id, tool="image_edit")
        try:
            args = ImageEditArgs(
                image_ref=image_ref,
                prompt=prompt,
                mask_ref=mask_ref,
                output_ref=output_ref,
                format=format,
                size=size,
                overwrite=overwrite,
                model=model,
                output_headers=output_headers,
            )
            if not args.prompt.strip():
                raise mcp_error(INVALID_ARGUMENT, "prompt is required")
            input_data = read_input(args.image_ref, self._settings)
            _validate_image_input(args.image_ref, input_data)
            image_size = _image_dimensions(input_data.data)
            mask_data: Optional[InputData] = None
            if args.mask_ref:
                mask_data = read_input(args.mask_ref, self._settings)
                _validate_image_input(args.mask_ref, mask_data)
                mask_size = _image_dimensions(mask_data.data)
                if mask_size != image_size:
                    raise mcp_error(INVALID_ARGUMENT, "mask dimensions must match source image")
            if args.size:
                parsed = _parse_image_size(args.size)
                if parsed != image_size:
                    raise mcp_error(INVALID_ARGUMENT, "size must match source image dimensions")
            output_format = _normalize_image_format(args.format)
            result = self._client.edit_image(
                image_bytes=input_data.data,
                prompt=args.prompt,
                mask_bytes=mask_data.data if mask_data else None,
                model_override=args.model,
                size=args.size,
                output_format=output_format,
                source_filename=_filename_from_ref(args.image_ref),
                mask_filename=_filename_from_ref(args.mask_ref) if args.mask_ref else None,
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
                "model": args.model or self._settings.openai_model_image_edit,
                "duration_ms": result.duration_ms,
                "format": output_format or "png",
                "size": args.size,
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
            self._log_info("image_edit end", request_id, tool="image_edit")

    def image_extract(
        self,
        image_ref: str,
        instruction: str,
        json_schema: Dict[str, Any],
        language: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract structured data from an image with schema validation."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("image_extract start", request_id, tool="image_extract")
        try:
            args = ImageExtractArgs(
                image_ref=image_ref,
                instruction=instruction,
                json_schema=json_schema,
                language=language,
                max_output_tokens=max_output_tokens,
                model=model,
            )
            if not args.instruction.strip():
                raise mcp_error(INVALID_ARGUMENT, "instruction is required")
            input_data = read_input(args.image_ref, self._settings)
            _validate_image_input(args.image_ref, input_data)
            result = self._client.extract_image(
                image_bytes=input_data.data,
                instruction=args.instruction,
                json_schema=args.json_schema,
                model_override=args.model,
                language=args.language,
                max_output_tokens=args.max_output_tokens,
            )
            try:
                jsonschema_validate(result.json_data, args.json_schema)
            except JsonSchemaValidationError as exc:
                raise mcp_error(SCHEMA_VALIDATION_FAILED, "Model output failed JSON schema validation", exc)
            metadata: Dict[str, Any] = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_vision,
                "duration_ms": result.duration_ms,
                "input_bytes": input_data.size,
                "json": result.json_data,
            }
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
            self._log_info("image_extract end", request_id, tool="image_extract")

    def image_to_spec(
        self,
        image_ref: str,
        target_format: str,
        instruction: Optional[str] = None,
        output_ref: Optional[str] = None,
        overwrite: bool = False,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Convert an image into a structured textual spec."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("image_to_spec start", request_id, tool="image_to_spec")
        try:
            args = ImageToSpecArgs(
                image_ref=image_ref,
                target_format=target_format,
                instruction=instruction,
                output_ref=output_ref,
                overwrite=overwrite,
                model=model,
                output_headers=output_headers,
            )
            if args.target_format not in IMAGE_SPEC_FORMATS:
                raise mcp_error(INVALID_ARGUMENT, f"Unsupported target_format: {args.target_format}")
            input_data = read_input(args.image_ref, self._settings)
            _validate_image_input(args.image_ref, input_data)
            result = self._client.image_to_spec(
                image_bytes=input_data.data,
                instruction=args.instruction,
                target_format=args.target_format,
                model_override=args.model,
                max_output_tokens=None,
            )
            outputs: List[OutputInfo] = []
            metadata: Dict[str, Any] = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_vision,
                "duration_ms": result.duration_ms,
                "format": args.target_format,
                "input_bytes": input_data.size,
            }
            if args.output_ref:
                _kind, bytes_written, sha256, path_or_url = write_output_text(
                    ref=args.output_ref,
                    text=result.text,
                    settings=self._settings,
                    overwrite=args.overwrite,
                    headers=args.output_headers,
                )
                output_info = OutputInfo(
                    kind="text",
                    path_or_url=path_or_url,
                    mime_type="text/plain; charset=utf-8",
                    bytes_written=bytes_written,
                    sha256=sha256,
                )
                outputs.append(output_info)
            else:
                metadata["text"] = result.text
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
            self._log_info("image_to_spec end", request_id, tool="image_to_spec")

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

    def audio_analyze(
        self,
        audio_ref: str,
        instruction: str,
        response_format: str = "text",
        json_schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze audio content and return text or schema-validated JSON."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("audio_analyze start", request_id, tool="audio_analyze")
        try:
            args = AudioAnalyzeArgs(
                audio_ref=audio_ref,
                instruction=instruction,
                response_format=response_format,
                json_schema=json_schema,
                model=model,
            )
            if not args.instruction.strip():
                raise mcp_error(INVALID_ARGUMENT, "instruction is required")
            if args.response_format not in {"text", "json"}:
                raise mcp_error(INVALID_ARGUMENT, "response_format must be text or json")
            if args.response_format == "json" and not args.json_schema:
                raise mcp_error(INVALID_ARGUMENT, "json_schema is required for json responses")
            input_data = read_input(args.audio_ref, self._settings)
            _validate_audio_input(args.audio_ref, input_data)
            source_filename = _filename_from_ref(args.audio_ref)
            result = self._client.analyze_audio(
                audio_bytes=input_data.data,
                instruction=args.instruction,
                model_override=args.model,
                response_format=args.response_format,
                json_schema=args.json_schema,
                source_filename=source_filename,
            )
            metadata: Dict[str, Any] = {
                "request_id": request_id,
                "model": args.model or self._settings.openai_model_audio_analyze,
                "duration_ms": result.duration_ms,
                "input_bytes": input_data.size,
            }
            if args.response_format == "json":
                json_schema = args.json_schema
                if json_schema is None or result.json_data is None:
                    raise mcp_error(INVALID_ARGUMENT, "json_schema is required for json responses")
                try:
                    jsonschema_validate(result.json_data, json_schema)
                except JsonSchemaValidationError as exc:
                    raise mcp_error(SCHEMA_VALIDATION_FAILED, "Model output failed JSON schema validation", exc)
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
            self._log_info("audio_analyze end", request_id, tool="audio_analyze")

    def audio_transform(
        self,
        audio_ref: str,
        instruction: str,
        output_ref: str,
        voice: Optional[str] = None,
        format: Optional[str] = None,
        overwrite: bool = False,
        model: Optional[str] = None,
        output_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Transform speech audio based on an instruction and write output audio."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("audio_transform start", request_id, tool="audio_transform")
        try:
            args = AudioTransformArgs(
                audio_ref=audio_ref,
                instruction=instruction,
                output_ref=output_ref,
                voice=voice,
                format=format,
                overwrite=overwrite,
                model=model,
                output_headers=output_headers,
            )
            if not args.instruction.strip():
                raise mcp_error(INVALID_ARGUMENT, "instruction is required")
            input_data = read_input(args.audio_ref, self._settings)
            _validate_audio_input(args.audio_ref, input_data)
            output_format = _normalize_tts_format(args.format)
            result = self._client.transform_audio(
                audio_bytes=input_data.data,
                instruction=args.instruction,
                model_override=args.model,
                voice=args.voice,
                format=output_format,
                source_filename=_filename_from_ref(args.audio_ref),
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
                "model": args.model or self._settings.openai_model_audio_transform,
                "duration_ms": result.duration_ms,
                "format": output_format or "mp3",
                "voice": args.voice,
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
            self._log_info("audio_transform end", request_id, tool="audio_transform")

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

    def multimodal_chain(
        self,
        steps: List[Dict[str, Any]],
        final_output_ref: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Execute a deterministic sequence of multimodal steps."""
        request_id = self._new_request_id()
        warnings: List[str] = []
        self._log_info("multimodal_chain start", request_id, tool="multimodal_chain")
        try:
            args = MultimodalChainArgs(
                steps=steps,
                final_output_ref=final_output_ref,
                overwrite=overwrite,
            )
            outputs: List[OutputInfo] = []
            resolved: Dict[str, Any] = {}
            step_summaries: List[Dict[str, Any]] = []
            last_result: Optional[Dict[str, Any]] = None
            for index, step in enumerate(args.steps, start=1):
                tool_name = step.tool
                tool_fn = self._chain_tool_handlers().get(tool_name)
                if tool_fn is None:
                    raise mcp_error(INVALID_ARGUMENT, f"Unsupported tool in chain: {tool_name}")
                resolved_args = _resolve_chain_args(step.args, resolved)
                if not isinstance(resolved_args, dict):
                    raise mcp_error(INVALID_ARGUMENT, f"Step args must be an object for {tool_name}")
                result = tool_fn(**resolved_args)
                last_result = result
                warnings.extend(result.get("warnings", []))
                step_summaries.append({"index": index, "tool": tool_name, "ok": result.get("ok", False)})
                if not result.get("ok", False):
                    error = result.get("error") or {}
                    message = error.get("message", "Chain step failed")
                    code = error.get("code", "UNKNOWN")
                    error_info = ErrorInfo(
                        code=CHAIN_STEP_FAILED,
                        message=f"Step {index} ({tool_name}) failed: {code} {message}",
                    )
                    return ToolResult(
                        ok=False,
                        outputs=outputs,
                        metadata={"request_id": request_id, "steps": step_summaries},
                        warnings=warnings,
                        error=error_info,
                    ).model_dump()
                for output in result.get("outputs", []):
                    outputs.append(OutputInfo(**output))
                if step.outputs_as:
                    resolved[step.outputs_as] = result
            if args.final_output_ref:
                if last_result is None:
                    raise mcp_error(INVALID_ARGUMENT, "No steps executed for final_output_ref")
                final_info = _write_chain_final_output(
                    args.final_output_ref,
                    last_result,
                    settings=self._settings,
                    overwrite=args.overwrite,
                )
                outputs.append(final_info)
            return ToolResult(
                ok=True,
                outputs=outputs,
                metadata={"request_id": request_id, "steps": step_summaries},
                warnings=warnings,
            ).model_dump()
        except ValidationError as exc:
            return self._error_result(request_id, exc, warnings)
        except MCPError as exc:
            return self._error_result(request_id, exc, warnings)
        except Exception as exc:
            return self._error_result(request_id, exc, warnings)
        finally:
            self._log_info("multimodal_chain end", request_id, tool="multimodal_chain")

    def _chain_tool_handlers(self) -> Dict[str, Any]:
        return {
            "image_generate": self.image_generate,
            "image_analyze": self.image_analyze,
            "image_edit": self.image_edit,
            "image_extract": self.image_extract,
            "image_to_spec": self.image_to_spec,
            "audio_transcribe": self.audio_transcribe,
            "audio_analyze": self.audio_analyze,
            "audio_transform": self.audio_transform,
            "audio_tts": self.audio_tts,
        }

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
        description="Edit or inpaint an image and write the result to the output reference.",
    )(service.image_edit)
    mcp.tool(
        description="Extract structured data from an image with schema validation.",
    )(service.image_extract)
    mcp.tool(
        description="Convert an image into a structured textual spec.",
    )(service.image_to_spec)
    mcp.tool(
        description="Transcribe audio to text and optionally write the transcript to a file.",
    )(service.audio_transcribe)
    mcp.tool(
        description="Analyze audio content and return text or schema-validated JSON.",
    )(service.audio_analyze)
    mcp.tool(
        description="Transform speech audio based on an instruction and write output audio.",
    )(service.audio_transform)
    mcp.tool(
        description="Generate speech audio from text and write it to the output reference.",
    )(service.audio_tts)
    mcp.tool(
        description="Execute a deterministic sequence of multimodal steps.",
    )(service.multimodal_chain)
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


def _filename_from_ref(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    parsed = urlparse(ref)
    path = parsed.path if parsed.scheme else ref
    name = Path(path).name
    return name or None


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        dimensions = _png_dimensions(image_bytes)
        if dimensions is None:
            raise mcp_error(INTERNAL_ERROR, "Pillow is required for non-PNG image dimension checks", exc)
        return dimensions
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.size


def _parse_image_size(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise mcp_error(INVALID_ARGUMENT, f"Invalid size format: {value}")
    width_str, height_str = value.split("x", 1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise mcp_error(INVALID_ARGUMENT, f"Invalid size format: {value}", exc)
    if width <= 0 or height <= 0:
        raise mcp_error(INVALID_ARGUMENT, f"Invalid size format: {value}")
    return width, height


def _png_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    if len(image_bytes) < 24:
        return None
    signature = image_bytes[:8]
    if signature != b"\x89PNG\r\n\x1a\n":
        return None
    ihdr_type = image_bytes[12:16]
    if ihdr_type != b"IHDR":
        return None
    width = int.from_bytes(image_bytes[16:20], "big")
    height = int.from_bytes(image_bytes[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _resolve_chain_args(value: Any, resolved: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"$ref"}:
            return _resolve_chain_ref(value["$ref"], resolved)
        return {key: _resolve_chain_args(val, resolved) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_chain_args(item, resolved) for item in value]
    return value


def _resolve_chain_ref(ref: str, resolved: Dict[str, Any]) -> Any:
    if not isinstance(ref, str) or not ref:
        raise mcp_error(INVALID_ARGUMENT, "Invalid reference in chain arguments")
    tokens = _tokenize_ref(ref)
    if not tokens:
        raise mcp_error(INVALID_ARGUMENT, "Invalid reference in chain arguments")
    root = tokens.pop(0)
    if root not in resolved:
        raise mcp_error(INVALID_ARGUMENT, f"Unknown reference: {root}")
    current: Any = resolved[root]
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise mcp_error(INVALID_ARGUMENT, f"Invalid reference index in {ref}")
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                raise mcp_error(INVALID_ARGUMENT, f"Invalid reference field in {ref}")
            current = current[token]
    return current


def _tokenize_ref(ref: str) -> List[Any]:
    tokens: List[Any] = []
    buffer = ""
    index = 0
    while index < len(ref):
        char = ref[index]
        if char == ".":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            index += 1
            continue
        if char == "[":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            end = ref.find("]", index)
            if end == -1:
                raise mcp_error(INVALID_ARGUMENT, f"Invalid reference syntax: {ref}")
            idx_str = ref[index + 1:end]
            if not idx_str.isdigit():
                raise mcp_error(INVALID_ARGUMENT, f"Invalid reference index: {ref}")
            tokens.append(int(idx_str))
            index = end + 1
            continue
        buffer += char
        index += 1
    if buffer:
        tokens.append(buffer)
    return tokens


def _write_chain_final_output(
    ref: str,
    last_result: Dict[str, Any],
    settings: Settings,
    overwrite: bool,
) -> OutputInfo:
    metadata = last_result.get("metadata", {})
    if "json" in metadata:
        payload = json.dumps(metadata["json"], ensure_ascii=True, sort_keys=True, indent=2)
        data = payload.encode("utf-8")
        mime_type = "application/json; charset=utf-8"
        _kind, bytes_written, sha256, path_or_url = write_output_bytes(
            ref=ref,
            data=data,
            mime_type=mime_type,
            settings=settings,
            overwrite=overwrite,
        )
        return OutputInfo(
            kind="json",
            path_or_url=path_or_url,
            mime_type=mime_type,
            bytes_written=bytes_written,
            sha256=sha256,
        )
    if "text" in metadata:
        _kind, bytes_written, sha256, path_or_url = write_output_text(
            ref=ref,
            text=str(metadata["text"]),
            settings=settings,
            overwrite=overwrite,
        )
        return OutputInfo(
            kind="text",
            path_or_url=path_or_url,
            mime_type="text/plain; charset=utf-8",
            bytes_written=bytes_written,
            sha256=sha256,
        )
    raise mcp_error(INVALID_ARGUMENT, "final_output_ref requires text or json output from the last step")
