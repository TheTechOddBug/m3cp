# agents.md — Feature Spec: Multimodal MCP Server (OpenAI-only)

Status: Draft
Owner: (you)
Last updated: 2026-01-11
Language/runtime: Python 3.11+ (prefer 3.12)
MCP framework: FastMCP
Scope: MCP server providing file-oriented image/audio generation + interpretation tools using the latest OpenAI API interfaces.

## 0. Outcome

Implement an MCP server that exposes practical multimedia tools to MCP clients (e.g., IDE agents). The server is “file-first”: it reads inputs from file paths/URLs supplied by the client and writes outputs to file paths/URLs supplied by the client. The server maintains no repository concept, no workspace root, and no implicit output directory.

The server only integrates with OpenAI (no other providers). Use the latest OpenAI API surfaces: Responses API for vision analysis; Images API for image generation; Audio endpoints for STT and TTS.

## 1. Non-goals

Do not implement:

- Any notion of “project” or “repo” inside the server.
- Any background job queue, scheduler, or database persistence.
- Any custom UI, web frontend, or auth system beyond API key usage and basic allowlists.
- Realtime voice sessions (not needed yet). Ensure architecture allows adding later without refactor.

## 2. Operating model and assumptions

- The MCP server runs in an environment where it is permitted to read/write the files referenced by the client (local filesystem paths, and optionally HTTP(S) URLs if enabled).
- The client passes explicit, concrete input locations and explicit output destinations.
- The MCP server must not “guess” destinations, create nested output structures, or infer a workspace root.
- All tool calls must be deterministic in side effects: only read specified inputs; only write to specified output paths.

## 3. Tool surface (MCP)

Expose exactly these MCP tools (names are stable API):

1) `image_generate`
2) `image_analyze`
3) `audio_transcribe`
4) `audio_tts`

Each tool must accept either:

- local file paths (recommended), or
- HTTP(S) URLs, if `ENABLE_REMOTE_URLS=true`.

When writing output:

- allow local file path output
- optionally allow HTTP(S) upload only if the client provides a pre-signed URL and `ENABLE_PRESIGNED_UPLOADS=true` (implement PUT upload). Otherwise reject remote outputs.

Return a structured result containing:

- `ok: bool`
- `outputs: [{ kind, path_or_url, mime_type, bytes_written, sha256 }]` when applicable
- `metadata: {...}` (model used, timing, sizes)
- `warnings: [..]`
- `error: { code, message }` if `ok=false`

## 4. OpenAI integration requirements

Use the official OpenAI Python SDK and the newest interface patterns.

- Vision analysis: use the Responses API (vision-capable model) and send the image as input content. Return text and (optionally) structured JSON if requested.
- Image generation: use the Images API with a current image generation model. Return image bytes; write to output destination.
- Speech-to-text: use the current transcription model endpoint; return transcript (and optionally timestamps if supported by the selected model; otherwise omit).
- Text-to-speech: use the current TTS model endpoint; return audio bytes; write to output destination.

Mandatory environment variables:

- `OPENAI_API_KEY` (required)
Optional:
- `OPENAI_BASE_URL` (default SDK)
- `OPENAI_ORG_ID` (if needed)
- `OPENAI_PROJECT` (if needed)
- `OPENAI_MODEL_VISION` (default a sensible vision model)
- `OPENAI_MODEL_IMAGE` (default a sensible image model)
- `OPENAI_MODEL_STT` (default a sensible transcription model)
- `OPENAI_MODEL_TTS` (default a sensible TTS model)
- `ENABLE_REMOTE_URLS` (default false)
- `ENABLE_PRESIGNED_UPLOADS` (default false)
- `MAX_INPUT_BYTES` (default 25MB)
- `MAX_OUTPUT_BYTES` (default 25MB)
- `LOG_LEVEL` (default INFO)

Do not hardcode model names. Use defaults but allow override via env and per-tool arguments.

## 5. File and URL handling

Implement a small “FileRef” layer:

Inputs:

- If `input_ref` is a local path:
  - require it exists
  - require it is a file
  - enforce `MAX_INPUT_BYTES`
- If `input_ref` is an URL:
  - only allow if `ENABLE_REMOTE_URLS=true`
  - only allow `https://` (and optionally `http://` behind an explicit flag if you decide)
  - enforce `MAX_INPUT_BYTES` while streaming
  - validate content-type where possible

Outputs:

- Local output path:
  - create parent directories if they exist? No: create parent dirs only if explicitly allowed by `ALLOW_MKDIR=true` (default false). If false and parent missing, fail.
  - never overwrite unless `overwrite=true` is passed.
- Pre-signed upload URL:
  - only allow if `ENABLE_PRESIGNED_UPLOADS=true`
  - support PUT with required headers
  - return the upload URL as output reference (or the provided final URL, if the client supplies both)

Always compute SHA-256 for outputs and return it.

Supported formats:

- Images: png, jpg/jpeg, webp (accept). Output default png unless caller specifies.
- Audio input: wav, mp3, m4a, ogg/opus (accept). Transcribe accepts whatever OpenAI endpoint supports; validate by extension + sniff mime.
- Audio output (TTS): mp3, wav, opus. Default mp3 unless caller specifies.

## 6. Tool specifications

### 6.1 `image_generate`

Purpose: Generate an image from a prompt and write it to the client-specified destination.

Inputs (tool args):

- `prompt: str` (required)
- `output_ref: str` (required) — local path or pre-signed URL
- `size: str` (optional) — e.g. "1024x1024", "1536x1024" (validate against allowed set)
- `background: str` (optional) — "transparent" or "opaque" (if supported by model; otherwise warn)
- `quality: str` (optional) — "standard" / "high" (validate; map to API if supported)
- `format: str` (optional) — "png" | "jpeg" | "webp"
- `overwrite: bool` (optional, default false)
- `seed: int` (optional) — only if supported; otherwise ignore with warning
- `safety: dict` (optional) — passthrough flags if supported; otherwise ignore with warning

Behavior:

- Call OpenAI image generation endpoint.
- Decode result to bytes.
- Write to output destination according to policy.
- Return output metadata + sha256.

Failure modes:

- Oversize output: reject if > MAX_OUTPUT_BYTES.
- Output exists and overwrite=false: reject.
- Remote output not enabled: reject.

### 6.2 `image_analyze`

Purpose: Interpret an image from a client-specified location and return text or JSON.

Inputs:

- `image_ref: str` (required) — local path or URL
- `instruction: str` (required) — what to do (caption, OCR-like extraction, Q&A, UI analysis)
- `response_format: str` (optional) — "text" (default) or "json"
- `json_schema: dict` (optional) — required if response_format="json"
- `max_output_tokens: int` (optional)
- `detail: str` (optional) — "low" | "high" if supported
- `language: str` (optional) — BCP-47 code; used to bias output language

Behavior:

- Read image bytes.
- Call OpenAI Responses API with image+instruction.
- If `response_format="json"`, enforce schema-conformant JSON (fail if cannot be parsed/validated).
- Return analysis result (not written to disk).

Security:

- No hidden file reads. Only read `image_ref`.

### 6.3 `audio_transcribe`

Purpose: Transcribe spoken audio to text.

Inputs:

- `audio_ref: str` (required)
- `language: str` (optional) — BCP-47 or ISO language code (pass if supported)
- `prompt: str` (optional) — domain vocabulary hints
- `timestamps: bool` (optional, default false) — best-effort; warn if unsupported
- `diarize: bool` (optional, default false) — best-effort; warn if unsupported
- `output_ref: str` (optional) — if provided, write transcript text to file (UTF-8)
- `overwrite: bool` (optional, default false)

Behavior:

- Read audio bytes.
- Call OpenAI transcription endpoint with selected model.
- Produce `transcript_text`.
- If output_ref is provided, write transcript to destination, else return transcript inline.
- If timestamps/diarize requested and endpoint doesn’t support, return transcript anyway and add warning.

### 6.4 `audio_tts`

Purpose: Convert text into speech audio and write to destination.

Inputs:

- `text: str` (required)
- `output_ref: str` (required)
- `voice: str` (optional) — default voice
- `format: str` (optional) — "mp3" | "wav" | "opus"
- `speed: float` (optional) — clamp to allowed range if supported; else warn
- `overwrite: bool` (optional, default false)

Behavior:

- Call OpenAI TTS endpoint.
- Write bytes to output destination.
- Return output metadata + sha256.

## 7. Implementation constraints

- Use FastMCP to implement the server and tools.
- Provide an executable entrypoint: `python -m mcp_multimodal_server` or `mcp-multimodal-server`.
- Provide strict type hints (mypy clean).
- Provide structured logging (json lines preferred) including request_id.
- Provide timeouts and retries with exponential backoff for OpenAI calls.
- Avoid large in-memory buffers when downloading remote inputs; stream to temp file if needed.
- Use a dedicated temp directory; ensure cleanup.

## 8. Project structure

Create this repository layout:

- `multimodal_mcp/`
  - `__init__.py`
  - `main.py` (CLI + server startup)
  - `server.py` (FastMCP registration)
  - `openai_client.py` (OpenAI SDK wrapper; model selection; common call helpers)
  - `fileref.py` (read/write abstraction; URL fetch; pre-signed upload; hashing; size limits)
  - `schemas.py` (pydantic models for tool args + results)
  - `errors.py` (error codes + exception mapping)
  - `config.py` (env parsing)
- `tests/`
  - `test_fileref.py`
  - `test_tools_contracts.py` (mock OpenAI client)
- `README.md`
- `pyproject.toml` (poetry or uv; pick one and keep it minimal)
- `.env.example`

## 9. Error model (stable)

Define error codes (string constants) and use them consistently:

- `INVALID_ARGUMENT`
- `INPUT_NOT_FOUND`
- `INPUT_TOO_LARGE`
- `OUTPUT_TOO_LARGE`
- `OUTPUT_EXISTS`
- `OUTPUT_PARENT_MISSING`
- `REMOTE_URLS_DISABLED`
- `UPLOADS_DISABLED`
- `UNSUPPORTED_FORMAT`
- `OPENAI_ERROR`
- `TIMEOUT`
- `INTERNAL_ERROR`

Never return raw stack traces to clients. Log stack traces server-side.

## 10. Testing strategy

Unit tests must run offline without calling OpenAI:

- Mock OpenAI client wrapper to return fixed bytes/text.
- Test file read/write policies: overwrite behavior, size limits, sha256 correctness.
- Test JSON schema enforcement in `image_analyze` (valid and invalid outputs).

Provide at least one integration test behind an opt-in flag `RUN_LIVE_TESTS=1` that calls OpenAI, but ensure CI does not run it by default.

## 11. Documentation (README must include)

- What the server does (4 tools)
- How to run locally
- Env vars
- Example MCP tool invocations (pseudo-code, not tied to any specific client)
- Security notes: file permissions, URL enabling, pre-signed uploads

## 12. Deliverables

The code generation agent must produce:

- Complete Python implementation
- Minimal packaging to run as a module or console script
- Tests and README
- No placeholders in core logic (small TODOs allowed only for Realtime future extension)

Acceptance criteria:

- The server starts and registers the four tools.
- Each tool validates inputs, enforces limits, calls the OpenAI wrapper, and returns structured results.
- Outputs are written exactly where the client requested (or rejected cleanly).
- Offline tests pass.

Now implement it.
