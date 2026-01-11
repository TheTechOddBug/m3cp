from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openai import OpenAI

from .config import Settings
from .errors import INVALID_ARGUMENT, MCPError, OPENAI_ERROR, TIMEOUT, mcp_error


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
class TranscriptionResult:
    text: str
    segments: Optional[Any]
    duration_ms: int


@dataclass
class SpeechResult:
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
        image_payload: Dict[str, Any] = {
            "type": "input_image",
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        }
        if detail:
            image_payload["detail"] = detail
        content = [
            {"type": "input_text", "text": payload_instruction},
            image_payload,
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
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": max_output_tokens,
                "response_format": response_format_payload,
            }
            params = {key: value for key, value in params.items() if value is not None}
            client: Any = self._client
            response = client.responses.create(**params)
        except httpx.TimeoutException as exc:
            raise mcp_error(TIMEOUT, f"OpenAI request timed out: {str(exc)}", exc)
        except Exception as exc:
            error_msg = f"OpenAI response error: {type(exc).__name__} - {str(exc)}"
            raise mcp_error(OPENAI_ERROR, error_msg, exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        text = getattr(response, "output_text", None)
        if text is None:
            text = response.output[0].content[0].text  # type: ignore[attr-defined]
        json_data: Optional[Dict[str, Any]] = None
        if response_format == "json":
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
        item = response.data[0] if hasattr(response, "data") else response["data"][0]
        if isinstance(item, dict):
            b64_val = item.get("b64_json")
            url_val = item.get("url")
        else:
            b64_val = getattr(item, "b64_json", None)
            url_val = getattr(item, "url", None)
        
        if b64_val:
            image_data = base64.b64decode(b64_val)
        elif url_val:
            # Download from URL for models like DALL-E-3
            url_response = httpx.get(url_val, timeout=30.0)
            url_response.raise_for_status()
            image_data = url_response.content
        else:
            error_details = f"b64_present={bool(b64_val)}, url_present={bool(url_val)}"
            raise mcp_error(OPENAI_ERROR, f"No image data in response ({error_details})")
        
        return ImageGenerationResult(data=image_data, duration_ms=duration_ms)

    @_retryable()
    def transcribe_audio(
        self,
        audio_bytes: bytes,
        model_override: Optional[str],
        language: Optional[str],
        prompt: Optional[str],
        timestamps: bool,
    ) -> TranscriptionResult:
        model = self._require_model(model_override, self._settings.openai_model_stt, "transcription")
        response_format = "verbose_json" if timestamps else "json"
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio"  # type: ignore[attr-defined]
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
