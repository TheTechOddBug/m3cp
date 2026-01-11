from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx

from .config import Settings
from .errors import (
    INVALID_ARGUMENT,
    INPUT_NOT_FOUND,
    INPUT_TOO_LARGE,
    INTERNAL_ERROR,
    OUTPUT_EXISTS,
    OUTPUT_PARENT_MISSING,
    OUTPUT_TOO_LARGE,
    REMOTE_URLS_DISABLED,
    UPLOADS_DISABLED,
    mcp_error,
)


@dataclass
class InputData:
    data: bytes
    mime_type: str
    size: int
    source: str


def is_url(ref: str) -> bool:
    parsed = urlparse(ref)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def read_input(ref: str, settings: Settings) -> InputData:
    if is_url(ref):
        if not settings.enable_remote_urls:
            raise mcp_error(REMOTE_URLS_DISABLED, "Remote URLs are disabled")
        return _read_remote(ref, settings)
    return _read_local(ref, settings)


def _read_local(ref: str, settings: Settings) -> InputData:
    path = Path(ref)
    if not path.exists() or not path.is_file():
        raise mcp_error(INPUT_NOT_FOUND, f"Input not found: {ref}")
    size = path.stat().st_size
    if size > settings.max_input_bytes:
        raise mcp_error(INPUT_TOO_LARGE, "Input exceeds MAX_INPUT_BYTES")
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return InputData(data=data, mime_type=mime_type, size=size, source=str(path))


def _read_remote(ref: str, settings: Settings) -> InputData:
    parsed = urlparse(ref)
    if parsed.scheme == "http" and not settings.allow_insecure_http:
        raise mcp_error(INVALID_ARGUMENT, "Insecure HTTP URLs are disabled")
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    size = 0
    mime_type = "application/octet-stream"
    try:
        with httpx.stream("GET", ref, timeout=30.0, follow_redirects=True) as response:
            if response.status_code >= 400:
                raise mcp_error(INPUT_NOT_FOUND, f"Remote input unavailable: {ref}")
            content_type = response.headers.get("content-type")
            if content_type:
                mime_type = content_type.split(";")[0].strip()
            with tempfile.NamedTemporaryFile(delete=False, dir=settings.temp_dir) as tmp:
                tmp_path = Path(tmp.name)
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > settings.max_input_bytes:
                        raise mcp_error(INPUT_TOO_LARGE, "Input exceeds MAX_INPUT_BYTES")
                    tmp.write(chunk)
        if tmp_path is None:
            raise mcp_error(INTERNAL_ERROR, "Failed to download remote input")
        data = tmp_path.read_bytes()
        return InputData(data=data, mime_type=mime_type, size=size, source=ref)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def write_output_bytes(
    ref: str,
    data: bytes,
    mime_type: str,
    settings: Settings,
    overwrite: bool,
    headers: Optional[Dict[str, str]] = None,
) -> tuple[str, int, str, str]:
    if len(data) > settings.max_output_bytes:
        raise mcp_error(OUTPUT_TOO_LARGE, "Output exceeds MAX_OUTPUT_BYTES")
    sha256 = hashlib.sha256(data).hexdigest()
    if is_url(ref):
        if not settings.enable_presigned_uploads:
            raise mcp_error(UPLOADS_DISABLED, "Presigned uploads are disabled")
        parsed = urlparse(ref)
        if parsed.scheme == "http" and not settings.allow_insecure_http:
            raise mcp_error(INVALID_ARGUMENT, "Insecure HTTP URLs are disabled")
        return _upload_remote(ref, data, mime_type, sha256, headers)
    return _write_local(ref, data, mime_type, sha256, settings, overwrite)


def _upload_remote(
    ref: str,
    data: bytes,
    mime_type: str,
    sha256: str,
    headers: Optional[Dict[str, str]],
) -> tuple[str, int, str, str]:
    upload_headers: Dict[str, str] = {}
    if headers:
        upload_headers.update(headers)
    if mime_type and "content-type" not in {key.lower() for key in upload_headers}:
        upload_headers["Content-Type"] = mime_type
    response = httpx.put(ref, content=data, headers=upload_headers, timeout=30.0)
    if response.status_code >= 400:
        raise mcp_error(INTERNAL_ERROR, f"Upload failed with status {response.status_code}")
    return "remote", len(data), sha256, ref


def _write_local(
    ref: str,
    data: bytes,
    mime_type: str,
    sha256: str,
    settings: Settings,
    overwrite: bool,
) -> tuple[str, int, str, str]:
    path = Path(ref)
    if path.exists():
        if path.is_dir():
            raise mcp_error(OUTPUT_EXISTS, f"Output path is a directory: {ref}")
        if not overwrite:
            raise mcp_error(OUTPUT_EXISTS, f"Output exists: {ref}")
    parent = path.parent
    if not parent.exists():
        if settings.allow_mkdir:
            parent.mkdir(parents=True, exist_ok=True)
        else:
            raise mcp_error(OUTPUT_PARENT_MISSING, f"Output parent missing: {parent}")
    path.write_bytes(data)
    return "file", len(data), sha256, str(path)


def write_output_text(
    ref: str,
    text: str,
    settings: Settings,
    overwrite: bool,
    headers: Optional[Dict[str, str]] = None,
) -> tuple[str, int, str, str]:
    data = text.encode("utf-8")
    return write_output_bytes(
        ref=ref,
        data=data,
        mime_type="text/plain; charset=utf-8",
        settings=settings,
        overwrite=overwrite,
        headers=headers,
    )
