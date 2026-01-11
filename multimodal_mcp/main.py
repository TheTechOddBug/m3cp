from __future__ import annotations

import sys

from .config import load_settings
from .logging_utils import setup_logging
from .server import build_server


def main() -> None:
    settings = load_settings()
    logger = setup_logging(settings.log_level)
    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is required", extra={"request_id": "startup"})
        sys.exit(1)
    mcp = build_server(settings=settings, logger=logger)
    mcp.run()


if __name__ == "__main__":
    main()
