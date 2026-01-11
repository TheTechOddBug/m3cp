from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

INVALID_ARGUMENT = "INVALID_ARGUMENT"
INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"
OUTPUT_EXISTS = "OUTPUT_EXISTS"
OUTPUT_PARENT_MISSING = "OUTPUT_PARENT_MISSING"
REMOTE_URLS_DISABLED = "REMOTE_URLS_DISABLED"
UPLOADS_DISABLED = "UPLOADS_DISABLED"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
OPENAI_ERROR = "OPENAI_ERROR"
TIMEOUT = "TIMEOUT"
INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class MCPError(Exception):
    code: str
    message: str
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def mcp_error(code: str, message: str, cause: Optional[BaseException] = None) -> MCPError:
    return MCPError(code=code, message=message, cause=cause)
