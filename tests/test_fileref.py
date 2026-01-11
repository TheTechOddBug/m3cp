from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from multimodal_mcp.config import Settings
from multimodal_mcp.errors import (
    INPUT_NOT_FOUND,
    INPUT_TOO_LARGE,
    OUTPUT_EXISTS,
    OUTPUT_PARENT_MISSING,
    MCPError,
)
from multimodal_mcp.fileref import read_input, write_output_bytes, write_output_text


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults = {
        "openai_api_key": "test",
        "openai_base_url": None,
        "openai_org_id": None,
        "openai_project": None,
        "openai_model_vision": "vision",
        "openai_model_image": "image",
        "openai_model_stt": "stt",
        "openai_model_tts": "tts",
        "enable_remote_urls": False,
        "enable_presigned_uploads": False,
        "allow_insecure_http": False,
        "allow_mkdir": False,
        "max_input_bytes": 5,
        "max_output_bytes": 8,
        "log_level": "INFO",
        "temp_dir": tmp_path / "temp",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_read_input_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(MCPError) as exc:
        read_input(str(tmp_path / "missing.txt"), settings)
    assert exc.value.code == INPUT_NOT_FOUND


def test_read_input_too_large(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_input_bytes=3)
    path = tmp_path / "input.txt"
    path.write_bytes(b"too-big")
    with pytest.raises(MCPError) as exc:
        read_input(str(path), settings)
    assert exc.value.code == INPUT_TOO_LARGE


def test_write_output_existing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_output_bytes=20)
    path = tmp_path / "output.bin"
    path.write_bytes(b"hello")
    with pytest.raises(MCPError) as exc:
        write_output_bytes(
            ref=str(path),
            data=b"world",
            mime_type="application/octet-stream",
            settings=settings,
            overwrite=False,
        )
    assert exc.value.code == OUTPUT_EXISTS


def test_write_output_parent_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_output_bytes=20)
    path = tmp_path / "missing" / "output.bin"
    with pytest.raises(MCPError) as exc:
        write_output_bytes(
            ref=str(path),
            data=b"world",
            mime_type="application/octet-stream",
            settings=settings,
            overwrite=False,
        )
    assert exc.value.code == OUTPUT_PARENT_MISSING


def test_write_output_sha256(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_output_bytes=20, allow_mkdir=True)
    path = tmp_path / "nested" / "output.txt"
    data = b"hello"
    _kind, bytes_written, sha256, path_or_url = write_output_bytes(
        ref=str(path),
        data=data,
        mime_type="text/plain",
        settings=settings,
        overwrite=False,
    )
    assert bytes_written == len(data)
    assert path_or_url == str(path)
    assert sha256 == hashlib.sha256(data).hexdigest()


def test_write_output_text(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_output_bytes=50, allow_mkdir=True)
    path = tmp_path / "nested" / "output.txt"
    _kind, bytes_written, sha256, path_or_url = write_output_text(
        ref=str(path),
        text="hello",
        settings=settings,
        overwrite=False,
    )
    assert bytes_written == 5
    assert path_or_url == str(path)
    assert sha256 == hashlib.sha256(b"hello").hexdigest()
