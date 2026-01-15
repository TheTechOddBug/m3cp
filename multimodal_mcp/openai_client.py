from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openai import OpenAI

from .config import Settings
from .errors import (
    INVALID_ARGUMENT,
    MCPError,
    OPENAI_ERROR,
    TIMEOUT,
    UNSUPPORTED_FORMAT,
    UNSUPPORTED_TRANSFORMATION,
    mcp_error,
)


def _openai_retry_exceptions() -> tuple[type[BaseException], ...]:
    exceptions: list[type[BaseException]] = [httpx.TimeoutException, httpx.HTTPError]
    try:
        import openai as openai_module

        for name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "APIError"):
            exc = getattr(openai_module, name, None)
            if exc is not None:
                exceptions.append(exc)
    except Exception:
        pass
    return tuple(exceptions)


_RETRYABLE = _openai_retry_exceptions()


def _retryable():
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(_RETRYABLE),
    )


@dataclass
class ImageAnalysisResult:
    text: str
    json_data: Optional[Dict[str, Any]]
    duration_ms: int


@dataclass
class ImageGenerationResult:
    data: bytes
    duration_ms: int


@dataclass
class ImageEditResult:
    data: bytes
    duration_ms: int


@dataclass
class TranscriptionResult:
    text: str
    segments: Optional[Any]
    duration_ms: int


@dataclass
class SpeechResult:
    data: bytes
    duration_ms: int


@dataclass
class ImageExtractResult:
    json_data: Dict[str, Any]
    duration_ms: int


@dataclass
class ImageSpecResult:
    text: str
    duration_ms: int


@dataclass
class AudioAnalysisResult:
    text: str
    json_data: Optional[Dict[str, Any]]
    duration_ms: int


@dataclass
class AudioTransformResult:
    data: bytes
    duration_ms: int


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise mcp_error(INVALID_ARGUMENT, "OPENAI_API_KEY is required")
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            organization=settings.openai_org_id,
            project=settings.openai_project,
            timeout=90.0,  # Image generation can take longer
            max_retries=0,
        )

    def _require_model(self, override: Optional[str], default: Optional[str], label: str) -> str:
        model = override or default
        if not model:
            raise mcp_error(INVALID_ARGUMENT, f"Model not configured for {label}")
        return model

    @_retryable()
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
        model = self._require_model(model_override, self._settings.openai_model_vision, "vision")
        payload_instruction = instruction
        if language:
            payload_instruction = f"Respond in {language}. {instruction}"
        
        # Use standard OpenAI Chat Completions API with vision
        image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": payload_instruction},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": detail or "auto"
                }
            }
        ]
        
        response_format_payload: Optional[Dict[str, Any]] = None
        if response_format == "json":
            if not json_schema:
                raise mcp_error(INVALID_ARGUMENT, "json_schema is required for JSON responses")
            response_format_payload = {
                "type": "json_schema",
                "json_schema": {
                    "name": "image_analysis",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        
        started = time.monotonic()
        try:
            params: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_output_tokens,
            }
            if response_format_payload:
                params["response_format"] = response_format_payload
            params = {key: value for key, value in params.items() if value is not None}
            client: Any = self._client
            response = client.chat.completions.create(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI response error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        
        text = response.choices[0].message.content
        json_data: Optional[Dict[str, Any]] = None
        if response_format == "json" and text:
            try:
                json_data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise mcp_error(OPENAI_ERROR, "Model output was not valid JSON", exc)
        return ImageAnalysisResult(text=text, json_data=json_data, duration_ms=duration_ms)

    @_retryable()
    def generate_image(
        self,
        prompt: str,
        model_override: Optional[str],
        size: Optional[str],
        background: Optional[str],
        quality: Optional[str],
        output_format: Optional[str],
    ) -> ImageGenerationResult:
        model = self._require_model(model_override, self._settings.openai_model_image, "image")
        started = time.monotonic()
        params: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "background": background,
            "quality": quality,
        }
        # Request base64 directly to avoid a second fetch for image URLs.
        params["response_format"] = "b64_json"
        if output_format:
            params["output_format"] = output_format
        params = {key: value for key, value in params.items() if value is not None}
        client: Any = self._client
        try:
            response = client.images.generate(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            retry_params = _strip_unsupported_image_params(params, exc)
            if retry_params is None:
                error_msg = f"OpenAI image generation error: {type(exc).__name__} - {str(exc)}"
                raise mcp_error(OPENAI_ERROR, error_msg, exc)
            try:
                response = client.images.generate(**retry_params)
            except httpx.TimeoutException as retry_exc:
                raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(retry_exc)}", retry_exc)
            except Exception as retry_exc:
                error_msg = f"OpenAI image generation error: {type(retry_exc).__name__} - {str(retry_exc)}"
                raise mcp_error(OPENAI_ERROR, error_msg, retry_exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        
        # Handle both b64_json and URL responses
        image_data = _extract_image_data(response)
        return ImageGenerationResult(data=image_data, duration_ms=duration_ms)

    @_retryable()
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
        model = self._require_model(model_override, self._settings.openai_model_image_edit, "image_edit")
        image_file = io.BytesIO(image_bytes)
        image_file.name = source_filename or "image.png"  # type: ignore[attr-defined]
        mask_file = None
        if mask_bytes is not None:
            mask_file = io.BytesIO(mask_bytes)
            mask_file.name = mask_filename or "mask.png"  # type: ignore[attr-defined]
        params: Dict[str, Any] = {
            "model": model,
            "image": image_file,
            "prompt": prompt,
            "size": size,
            "output_format": output_format,
        }
        if _model_requires_response_format(model):
            params["response_format"] = "b64_json"
        if mask_file is not None:
            params["mask"] = mask_file
        params = {key: value for key, value in params.items() if value is not None}
        started = time.monotonic()
        client: Any = self._client
        try:
            response = _call_image_edit(client, params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" in message and "unknown" in message:
                retry_params = dict(params)
                retry_params.pop("response_format", None)
                try:
                    response = _call_image_edit(client, retry_params)
                except Exception as retry_exc:
                    error_msg = f"OpenAI image edit error: {type(retry_exc).__name__} - {str(retry_exc)}"
                    raise mcp_error(OPENAI_ERROR, error_msg, retry_exc)
                duration_ms = int((time.monotonic() - started) * 1000)
                image_data = _extract_image_data(response)
                return ImageEditResult(data=image_data, duration_ms=duration_ms)
            if mask_bytes is not None and "mask" in message and "support" in message:
                raise mcp_error(UNSUPPORTED_FORMAT, "Mask editing is not supported by the selected model", exc)
            error_msg = f"OpenAI image edit error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        image_data = _extract_image_data(response)
        return ImageEditResult(data=image_data, duration_ms=duration_ms)

    @_retryable()
    def extract_image(
        self,
        image_bytes: bytes,
        instruction: str,
        json_schema: Dict[str, Any],
        model_override: Optional[str],
        language: Optional[str],
        max_output_tokens: Optional[int],
    ) -> ImageExtractResult:
        model = self._require_model(model_override, self._settings.openai_model_vision, "vision")
        payload_instruction = instruction
        if language:
            payload_instruction = f"Respond in {language}. {instruction}"
        content = [
            {"type": "input_text", "text": payload_instruction},
            _response_image_content(image_bytes),
        ]
        response_format_payload = {
            "type": "json_schema",
            "json_schema": {
                "name": "image_extract",
                "schema": json_schema,
                "strict": True,
            },
        }
        started = time.monotonic()
        try:
            params: Dict[str, Any] = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "response_format": response_format_payload,
            }
            if max_output_tokens is not None:
                params["max_output_tokens"] = max_output_tokens
            response = self._client.responses.create(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI vision extract error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        json_data = _extract_response_json(response)
        if json_data is None:
            raise mcp_error(OPENAI_ERROR, "Model output was not valid JSON")
        return ImageExtractResult(json_data=json_data, duration_ms=duration_ms)

    @_retryable()
    def image_to_spec(
        self,
        image_bytes: bytes,
        instruction: Optional[str],
        target_format: str,
        model_override: Optional[str],
        max_output_tokens: Optional[int],
    ) -> ImageSpecResult:
        model = self._require_model(model_override, self._settings.openai_model_vision, "vision")
        prompt = f"Convert the image into {target_format}."
        if instruction:
            prompt = f"{prompt} {instruction}"
        content = [
            {"type": "input_text", "text": prompt},
            _response_image_content(image_bytes),
        ]
        started = time.monotonic()
        try:
            params: Dict[str, Any] = {
                "model": model,
                "input": [{"role": "user", "content": content}],
            }
            if max_output_tokens is not None:
                params["max_output_tokens"] = max_output_tokens
            response = self._client.responses.create(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI image spec error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        text = _extract_response_text(response)
        return ImageSpecResult(text=text, duration_ms=duration_ms)

    @_retryable()
    def analyze_audio(
        self,
        audio_bytes: bytes,
        instruction: str,
        model_override: Optional[str],
        response_format: str,
        json_schema: Optional[Dict[str, Any]],
        source_filename: Optional[str],
    ) -> AudioAnalysisResult:
        model = self._require_model(model_override, self._settings.openai_model_audio_analyze, "audio_analyze")
        transcript_text: Optional[str] = None
        if model.startswith("gpt-audio"):
            transcript = self.transcribe_audio(
                audio_bytes=audio_bytes,
                model_override=self._settings.openai_model_stt,
                language=None,
                prompt=None,
                timestamps=False,
                source_filename=source_filename,
            )
            transcript_text = transcript.text
        content = _audio_analysis_content_for_model(
            model,
            instruction,
            audio_bytes,
            source_filename,
            transcript_text,
        )
        response_format_payload: Optional[Dict[str, Any]] = None
        if response_format == "json":
            if not json_schema:
                raise mcp_error(INVALID_ARGUMENT, "json_schema is required for JSON responses")
            response_format_payload = {
                "type": "json_schema",
                "json_schema": {
                    "name": "audio_analysis",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        started = time.monotonic()
        try:
            if transcript_text:
                input_payload: Any = _audio_analysis_text_prompt(instruction, transcript_text)
            else:
                input_payload = [{"role": "user", "content": content}]
            params: Dict[str, Any] = {
                "model": model,
                "input": input_payload,
            }
            if response_format_payload is not None:
                params["response_format"] = response_format_payload
            try:
                response = self._client.responses.create(**params)
            except Exception as exc:
                fallback_model = self._settings.openai_model_vision
                if transcript_text and fallback_model and fallback_model != model and _should_fallback_audio_analysis(exc):
                    params["model"] = fallback_model
                    response = self._client.responses.create(**params)
                else:
                    raise
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            message = str(exc).lower()
            if "input_audio" in message and "invalid" in message:
                raise mcp_error(
                    UNSUPPORTED_FORMAT,
                    "Audio analysis model does not accept input_audio; use an audio-preview model",
                    exc,
                )
            if "input_file" in message and "unknown" in message:
                raise mcp_error(
                    UNSUPPORTED_FORMAT,
                    "Audio analysis model does not accept input_file; use an audio-preview model",
                    exc,
                )
            error_msg = f"OpenAI audio analysis error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        if response_format == "json":
            json_data = _extract_response_json(response)
            if json_data is None:
                raise mcp_error(OPENAI_ERROR, "Model output was not valid JSON")
            return AudioAnalysisResult(text="", json_data=json_data, duration_ms=duration_ms)
        text = _extract_response_text(response)
        return AudioAnalysisResult(text=text, json_data=None, duration_ms=duration_ms)

    @_retryable()
    def transform_audio(
        self,
        audio_bytes: bytes,
        instruction: str,
        model_override: Optional[str],
        voice: Optional[str],
        format: Optional[str],
        source_filename: Optional[str],
    ) -> AudioTransformResult:
        model = self._require_model(model_override, self._settings.openai_model_audio_transform, "audio_transform")
        audio_format = _audio_format_from_filename(source_filename)
        content = [
            {"type": "input_text", "text": instruction},
            _response_audio_content(audio_bytes, audio_format),
        ]
        started = time.monotonic()
        try:
            response = self._client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                modalities=["audio"],
                audio={
                    "voice": voice or "alloy",
                    "format": format or "mp3",
                },
            )
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            message = str(exc).lower()
            if "audio" in message and "support" in message:
                raise mcp_error(
                    UNSUPPORTED_TRANSFORMATION,
                    "Speech-to-speech is not supported by the selected model",
                    exc,
                )
            error_msg = f"OpenAI audio transform error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        audio_data = _extract_response_audio(response)
        return AudioTransformResult(data=audio_data, duration_ms=duration_ms)

    @_retryable()
    def transcribe_audio(
        self,
        audio_bytes: bytes,
        model_override: Optional[str],
        language: Optional[str],
        prompt: Optional[str],
        timestamps: bool,
        source_filename: Optional[str] = None,
    ) -> TranscriptionResult:
        model = self._require_model(model_override, self._settings.openai_model_stt, "transcription")
        response_format = "verbose_json" if timestamps else "json"
        audio_file = io.BytesIO(audio_bytes)
        # OpenAI API requires a filename with extension to detect audio format
        if source_filename:
            audio_file.name = source_filename  # type: ignore[attr-defined]
        else:
            audio_file.name = "audio.mp3"  # type: ignore[attr-defined]
        started = time.monotonic()
        try:
            response = self._client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                prompt=prompt,
                response_format=response_format,
            )
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI transcription error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        text = getattr(response, "text", None) or response.get("text")  # type: ignore[call-arg]
        segments = None
        if hasattr(response, "segments"):
            segments = response.segments
        elif isinstance(response, dict):
            segments = response.get("segments")
        return TranscriptionResult(text=text, segments=segments, duration_ms=duration_ms)

    @_retryable()
    def text_to_speech(
        self,
        text: str,
        model_override: Optional[str],
        voice: Optional[str],
        format: Optional[str],
        speed: Optional[float],
    ) -> SpeechResult:
        model = self._require_model(model_override, self._settings.openai_model_tts, "tts")
        started = time.monotonic()
        voice_to_use = voice or "alloy"  # Default voice if none provided
        params: Dict[str, Any] = {
            "model": model,
            "voice": voice_to_use,
            "input": text,
            "response_format": format,
            "speed": speed,
        }
        params = {key: value for key, value in params.items() if value is not None}
        try:
            client: Any = self._client
            response = client.audio.speech.create(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI TTS error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        data = _extract_binary(response)
        return SpeechResult(data=data, duration_ms=duration_ms)


def _extract_binary(response: Any) -> bytes:
    if isinstance(response, bytes):
        return response
    if hasattr(response, "content"):
        return response.content  # type: ignore[return-value]
    if hasattr(response, "read"):
        return response.read()  # type: ignore[return-value]
    return bytes(response)


def _extract_image_data(response: Any) -> bytes:
    item = response.data[0] if hasattr(response, "data") else response["data"][0]
    if isinstance(item, dict):
        b64_val = item.get("b64_json")
        url_val = item.get("url")
    else:
        b64_val = getattr(item, "b64_json", None)
        url_val = getattr(item, "url", None)

    if b64_val:
        return base64.b64decode(b64_val)
    if url_val:
        url_response = httpx.get(url_val, timeout=30.0)
        url_response.raise_for_status()
        return url_response.content
    error_details = f"b64_present={bool(b64_val)}, url_present={bool(url_val)}"
    raise mcp_error(OPENAI_ERROR, f"No image data in response ({error_details})")


def _model_requires_response_format(model: str) -> bool:
    return model.startswith("dall-e") or model == "gpt-image-1"


def _call_image_edit(client: Any, params: Dict[str, Any]) -> Any:
    images_client = client.images
    if hasattr(images_client, "edit"):
        return images_client.edit(**params)
    if hasattr(images_client, "edits"):
        return images_client.edits(**params)
    raise mcp_error(OPENAI_ERROR, "OpenAI client does not support image edits")


def _response_image_content(image_bytes: bytes) -> Dict[str, Any]:
    image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    return {"type": "input_image", "image_url": image_url}


def _response_audio_content(audio_bytes: bytes, audio_format: str) -> Dict[str, Any]:
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}


def _audio_analysis_content_for_model(
    model: str,
    instruction: str,
    audio_bytes: bytes,
    source_filename: Optional[str],
    transcript_text: Optional[str],
) -> List[Dict[str, Any]]:
    if model.startswith("gpt-audio"):
        if not transcript_text:
            raise mcp_error(OPENAI_ERROR, "Audio analysis requires a transcript for gpt-audio models")
        return [{"type": "input_text", "text": _audio_analysis_text_prompt(instruction, transcript_text)}]
    audio_format = _audio_format_from_filename(source_filename)
    return [
        {"type": "input_text", "text": instruction},
        _response_audio_content(audio_bytes, audio_format),
    ]


def _audio_format_from_filename(source_filename: Optional[str]) -> str:
    if not source_filename:
        return "mp3"
    suffix = source_filename.lower().rsplit(".", 1)
    ext = suffix[-1] if len(suffix) > 1 else ""
    mapping = {
        "wav": "wav",
        "mp3": "mp3",
        "m4a": "mp3",
        "ogg": "ogg",
        "opus": "opus",
    }
    return mapping.get(ext, "mp3")


def _audio_analysis_text_prompt(instruction: str, transcript_text: str) -> str:
    return f"{instruction}\n\nTranscript:\n{transcript_text}"


def _should_fallback_audio_analysis(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("invalid_request", "unexpected keyword", "issue with your request"))


def _extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text  # type: ignore[return-value]
    if isinstance(response, dict) and response.get("output_text"):
        return response["output_text"]
    output = response.output if hasattr(response, "output") else response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in {"output_text", "text"}:
                    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                    if text:
                        return text
    return ""


def _extract_response_json(response: Any) -> Optional[Dict[str, Any]]:
    output = response.output if hasattr(response, "output") else response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and "json" in part:
                    json_data = part.get("json")
                    if isinstance(json_data, dict):
                        return json_data
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in {"output_text", "text"}:
                    text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return None
    text = _extract_response_text(response)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _extract_response_audio(response: Any) -> bytes:
    output = response.output if hasattr(response, "output") else response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type in {"output_audio", "audio"}:
                    audio = part.get("audio") if isinstance(part, dict) else getattr(part, "audio", None)
                    if isinstance(audio, dict):
                        data = audio.get("data")
                        if data:
                            return base64.b64decode(data)
                if isinstance(part, dict) and "audio" in part:
                    audio = part.get("audio")
                    if isinstance(audio, dict) and audio.get("data"):
                        return base64.b64decode(audio["data"])
    if isinstance(response, dict) and response.get("audio"):
        audio = response.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            return base64.b64decode(audio["data"])
    raise mcp_error(OPENAI_ERROR, "No audio data in response")


def _strip_unsupported_image_params(
    params: Dict[str, Any],
    exc: BaseException,
) -> Optional[Dict[str, Any]]:
    message = str(exc).lower()
    if not any(token in message for token in ("unknown", "unrecognized", "unexpected", "invalid", "unsupported")):
        return None
    retry_params = dict(params)
    removed = False
    for name in ("output_format", "response_format", "background", "quality"):
        if name in retry_params and name in message:
            retry_params.pop(name, None)
            removed = True
    if not removed:
        for name in ("output_format", "response_format", "background", "quality"):
            if name in retry_params:
                retry_params.pop(name, None)
                removed = True
    return retry_params if removed else None
