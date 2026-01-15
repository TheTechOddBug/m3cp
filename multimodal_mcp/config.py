from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: Optional[str]
    openai_org_id: Optional[str]
    openai_project: Optional[str]
    openai_model_vision: Optional[str]
    openai_model_image: Optional[str]
    openai_model_image_edit: Optional[str]
    openai_model_stt: Optional[str]
    openai_model_tts: Optional[str]
    openai_model_audio_analyze: Optional[str]
    openai_model_audio_transform: Optional[str]
    enable_remote_urls: bool
    enable_presigned_uploads: bool
    allow_insecure_http: bool
    allow_mkdir: bool
    max_input_bytes: int
    max_output_bytes: int
    log_level: str
    temp_dir: Path


def load_settings() -> Settings:
    # Try to load .env from the current working directory first
    load_dotenv(override=False)
    
    # Also try from the workspace root (parent of this module)
    workspace_root = Path(__file__).parent.parent
    env_path = workspace_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    temp_root = Path(os.getenv("MCP_TEMP_DIR", tempfile.gettempdir()))
    temp_dir = temp_root / "multimodal_mcp"
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openai_org_id=os.getenv("OPENAI_ORG_ID"),
        openai_project=os.getenv("OPENAI_PROJECT"),
        openai_model_vision=os.getenv("OPENAI_MODEL_VISION"),
        openai_model_image=os.getenv("OPENAI_MODEL_IMAGE"),
        openai_model_image_edit=os.getenv("OPENAI_MODEL_IMAGE_EDIT"),
        openai_model_stt=os.getenv("OPENAI_MODEL_STT"),
        openai_model_tts=os.getenv("OPENAI_MODEL_TTS"),
        openai_model_audio_analyze=os.getenv("OPENAI_MODEL_AUDIO_ANALYZE"),
        openai_model_audio_transform=os.getenv("OPENAI_MODEL_AUDIO_TRANSFORM"),
        enable_remote_urls=_get_env_bool("ENABLE_REMOTE_URLS", False),
        enable_presigned_uploads=_get_env_bool("ENABLE_PRESIGNED_UPLOADS", False),
        allow_insecure_http=_get_env_bool("ALLOW_INSECURE_HTTP", False),
        allow_mkdir=_get_env_bool("ALLOW_MKDIR", False),
        max_input_bytes=_get_env_int("MAX_INPUT_BYTES", 25 * 1024 * 1024),
        max_output_bytes=_get_env_int("MAX_OUTPUT_BYTES", 25 * 1024 * 1024),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        temp_dir=temp_dir,
    )
