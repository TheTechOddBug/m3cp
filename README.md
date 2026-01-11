# Multimodal MCP Server

Multimodal MCP server exposing four file-oriented tools backed by the OpenAI API:

- `image_generate` - create an image from a prompt and write it to a client-specified destination.
- `image_analyze` - interpret an image and return text or schema-validated JSON.
- `audio_transcribe` - transcribe audio to text (optionally write transcript to a file).
- `audio_tts` - generate speech audio from text and write it to a client-specified destination.

The server is file-first: it only reads from explicit input paths/URLs and writes to explicit output paths/URLs.

## Run Locally

```bash
python -m mcp_multimodal_server
```

Or via the console script:

```bash
mcp-multimodal-server
```

## MCP Configuration (mcp.json)

Add the server to your MCP client's configuration. For Claude Desktop or other MCP-compatible clients, add to your `mcp.json`:

```json
{
  "mcpServers": {
    "multimodal": {
      "command": "python",
      "args": ["-m", "mcp_multimodal_server"],
      "cwd": "${workspace_folder}"
    }
  }
}
```

Or if using the console script:

```json
{
  "mcpServers": {
    "multimodal": {
      "command": "mcp-multimodal-server",
      "cwd": "/path/to/m3cp"
    }
  }
}
```

**Note:** The server will automatically load the `OPENAI_API_KEY` from the `.env` file in the workspace directory. Make sure your `.env` file contains:

```bash
OPENAI_API_KEY=your-openai-api-key
```

You can also override other environment variables in the `env` object if needed (e.g., `OPENAI_BASE_URL`, `ENABLE_REMOTE_URLS`, etc.).

## Environment Variables

Required:

- `OPENAI_API_KEY`

Optional configuration:

- `OPENAI_BASE_URL`
- `OPENAI_ORG_ID`
- `OPENAI_PROJECT`
- `OPENAI_MODEL_VISION`
- `OPENAI_MODEL_IMAGE`
- `OPENAI_MODEL_STT`
- `OPENAI_MODEL_TTS`
- `ENABLE_REMOTE_URLS` (default false)
- `ENABLE_PRESIGNED_UPLOADS` (default false)
- `ALLOW_INSECURE_HTTP` (default false)
- `ALLOW_MKDIR` (default false)
- `MAX_INPUT_BYTES` (default 25MB)
- `MAX_OUTPUT_BYTES` (default 25MB)
- `LOG_LEVEL` (default INFO)
- `MCP_TEMP_DIR` (default system temp dir)

Note: If the model environment variables are not set, pass a `model` override in the tool call.
The server loads a local `.env` file automatically if present.

## Example MCP Tool Calls (Pseudo-code)

```python
# image_generate
client.call_tool(
    "image_generate",
    {
        "prompt": "A watercolor map of a coastal city",
        "output_ref": "/tmp/city.png",
        "size": "1024x1024",
        "format": "png",
        "overwrite": True,
    },
)

# image_analyze
client.call_tool(
    "image_analyze",
    {
        "image_ref": "/tmp/city.png",
        "instruction": "Summarize the visual style",
        "response_format": "text",
    },
)

# audio_transcribe
client.call_tool(
    "audio_transcribe",
    {
        "audio_ref": "/tmp/meeting.wav",
        "timestamps": True,
        "output_ref": "/tmp/meeting.txt",
        "overwrite": True,
    },
)

# audio_tts
client.call_tool(
    "audio_tts",
    {
        "text": "Welcome to the demo!",
        "output_ref": "/tmp/welcome.mp3",
        "format": "mp3",
        "overwrite": True,
    },
)
```

## Security Notes

- The server only reads inputs explicitly provided by the client.
- Remote URLs are disabled unless `ENABLE_REMOTE_URLS=true`.
- Presigned uploads are disabled unless `ENABLE_PRESIGNED_UPLOADS=true`.
- Output directories are only created when `ALLOW_MKDIR=true`.
- Ensure the server has access only to the files and network locations you intend it to reach.
