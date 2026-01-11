from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OutputInfo(BaseModel):
    kind: str
    path_or_url: str
    mime_type: str
    bytes_written: int
    sha256: str


class ErrorInfo(BaseModel):
    code: str
    message: str


class ToolResult(BaseModel):
    ok: bool
    outputs: List[OutputInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ErrorInfo] = None


class ImageGenerateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    output_ref: str
    size: Optional[str] = None
    background: Optional[str] = None
    quality: Optional[str] = None
    format: Optional[str] = None
    overwrite: bool = False
    seed: Optional[int] = None
    safety: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    output_headers: Optional[Dict[str, str]] = None


class ImageAnalyzeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ref: str
    instruction: str
    response_format: str = "text"
    json_schema: Optional[Dict[str, Any]] = None
    max_output_tokens: Optional[int] = None
    detail: Optional[str] = None
    language: Optional[str] = None
    model: Optional[str] = None


class AudioTranscribeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_ref: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    timestamps: bool = False
    diarize: bool = False
    output_ref: Optional[str] = None
    overwrite: bool = False
    model: Optional[str] = None
    output_headers: Optional[Dict[str, str]] = None


class AudioTtsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    output_ref: str
    voice: Optional[str] = None
    format: Optional[str] = None
    speed: Optional[float] = None
    overwrite: bool = False
    model: Optional[str] = None
    output_headers: Optional[Dict[str, str]] = None
