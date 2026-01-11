# Multimodal MCP Server

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)
![MCP](https://img.shields.io/badge/MCP-Server-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FOSS Pluralism](https://img.shields.io/badge/FOSS-Pluralism-purple.svg)

A production-ready Model Context Protocol (MCP) server that brings OpenAI's multimodal capabilities—vision, image generation, speech-to-text, and text-to-speech—to any MCP-compatible client. Built with a file-first architecture for security and transparency, ensuring all operations use explicit input/output paths.

## Features

Multimodal MCP server exposing four file-oriented tools backed by the OpenAI API:

- `image_generate` - create an image from a prompt and write it to a client-specified destination.
- `image_analyze` - interpret an image and return text or schema-validated JSON.
- `audio_transcribe` - transcribe audio to text (optionally write transcript to a file).
- `audio_tts` - generate speech audio from text and write it to a client-specified destination.

The server is file-first: it only reads from explicit input paths/URLs and writes to explicit output paths/URLs.

![Tools](./mcp-tools.png)
> the image was created by the MCP server

[Audio description of the project](./mcp-tools.mp3)
> the audio file was created by the MCP server

## Run Locally

```bash
python -m multimodal_mcp.main
```

Or via the console script (after installing with `pip install -e .` or `uv pip install -e .`):

```bash
mcp-multimodal-server
```

## MCP Configuration (mcp.json)

Add the server to your MCP client's configuration. For Claude Desktop or other MCP-compatible clients, add to your `.vscode/mcp.json`:

```json
{
  "servers": {
    "multimodal_mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "${workspaceFolder}", "run", "multimodal_mcp_server.py"]
    }
  },
  "inputs": []
}
```

Or if you've installed the package and want to use the console script:

```json
{
  "servers": {
    "multimodal_mcp": {
      "type": "stdio",
      "command": "mcp-multimodal-server"
    }
  },
  "inputs": []
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

## Principles of Participation

Everyone is invited and welcome to contribute: open issues, propose pull requests, share ideas, or help improve documentation.  
Participation is open to all, regardless of background or viewpoint.  

This project follows the [FOSS Pluralism Manifesto](./FOSS_PLURALISM_MANIFESTO.md),  
which affirms respect for people, freedom to critique ideas, and space for diverse perspectives.  

## License and Copyright

Copyright (c) 2026, Iwan van der Kleijn

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.