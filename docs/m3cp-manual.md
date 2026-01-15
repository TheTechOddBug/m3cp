# MCP3 Manual - Multimodal MCP Server

This manual describes how to use the Multimodal MCP Server tools in IDE agents, orchestrators, and pipelines. It complements the README and focuses on tool semantics, constraints, and predictable behavior.

## Core operating model

- File-first: every tool reads explicit input refs and writes only to explicit output refs.
- Deterministic side effects: no hidden reads, no implicit outputs.
- OpenAI-only: all multimodal calls are routed to the OpenAI API.
- No realtime, streaming, or video functionality.

## File references

Inputs can be local files or HTTP(S) URLs when `ENABLE_REMOTE_URLS=true`.
Outputs can be local files or presigned upload URLs when `ENABLE_PRESIGNED_UPLOADS=true`.

Local output rules:
- Parent directories are created only when `ALLOW_MKDIR=true`.
- Outputs never overwrite unless `overwrite=true`.

## Tools

### image_generate
Generate an image from a prompt and write it to `output_ref`.

Key inputs:
- `prompt` (required)
- `output_ref` (required)
- `size`, `quality`, `background`, `format` (optional)

### image_analyze
Analyze an image with a natural language instruction.

Key inputs:
- `image_ref` (required)
- `instruction` (required)
- `response_format` (text or json)
- `json_schema` (required for json)

### image_edit
Edit or inpaint an image with an optional mask.

Key inputs:
- `image_ref` (required)
- `prompt` (required)
- `mask_ref` (optional, must match image dimensions)
- `output_ref` (required)

Constraints:
- Mask and image dimensions must match exactly.
- If `size` is provided, it must match the source image dimensions.

### image_extract
Extract structured JSON from an image with a strict schema.

Key inputs:
- `image_ref` (required)
- `instruction` (required)
- `json_schema` (required)

Behavior:
- JSON output is validated strictly against the provided schema.

### image_to_spec
Convert an image into a textual spec (diagram or document format).

Key inputs:
- `image_ref` (required)
- `target_format` (required: mermaid, plantuml, openapi, c4, markdown)
- `instruction` (optional)
- `output_ref` (optional)

If `output_ref` is omitted, the spec is returned inline.

### audio_transcribe
Transcribe audio to text with optional timestamps.

Key inputs:
- `audio_ref` (required)
- `output_ref` (optional)

### audio_analyze
Analyze audio content (tone, sentiment, dynamics, etc.).

Key inputs:
- `audio_ref` (required)
- `instruction` (required)
- `response_format` (text or json)
- `json_schema` (required for json)

### audio_transform
Transform speech-to-speech based on an instruction.

Key inputs:
- `audio_ref` (required)
- `instruction` (required)
- `output_ref` (required)
- `voice` and `format` (optional)

### audio_tts
Generate speech audio from text.

Key inputs:
- `text` (required)
- `output_ref` (required)
- `voice` and `format` (optional)

### multimodal_chain
Execute a deterministic, explicit sequence of tool calls.

Inputs:
```
{
  "steps": [
    {"tool": "image_analyze", "args": {"image_ref": "...", "instruction": "..."}, "outputs_as": "analysis"},
    {"tool": "audio_tts", "args": {"text": {"$ref": "analysis.metadata.text"}, "output_ref": "..."}}
  ],
  "final_output_ref": "optional"
}
```

Reference resolution:
- Use `{"$ref": "name.path"}` to reference prior outputs.
- `name` is the `outputs_as` value for a previous step.
- Path segments may include list indices: `outputs[0].path_or_url`.

Failure behavior:
- The chain stops on the first failed step and returns partial outputs.

`final_output_ref`:
- If provided, the chain writes the final step's inline `text` or `json` output to the given ref.

## Error codes (additions)

- `SCHEMA_VALIDATION_FAILED` for schema validation failures.
- `CHAIN_STEP_FAILED` when a chain step fails.
- `UNSUPPORTED_TRANSFORMATION` when speech-to-speech is unsupported by the model.
