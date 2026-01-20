# Implementation details: 

## Tool Registration in FastMCP

This  chapter explains the design decision for tool registration in the Multimodal MCP Server, specifically regarding the choice between functional registration and decorator-based registration patterns.

## Current Implementation: Functional Registration

The `build_server()` function uses **functional registration** where `mcp.tool()` is called manually to register service methods:

```python
def build_server(
    settings: Optional[Settings] = None,
    client: Optional[OpenAIClientProtocol] = None,
    logger: Optional[logging.Logger] = None,
) -> FastMCP:
    settings = settings or load_settings()
    logger = logger or setup_logging(settings.log_level)
    client = client or OpenAIClient(settings)
    service = ToolService(settings, client, logger)
    mcp = FastMCP("multimodal-mcp")
    
    mcp.tool(
        description="Generate an image from a prompt and write it to the output reference.",
    )(service.image_generate)
    
    mcp.tool(
        description="Analyze an image and return text or schema-validated JSON.",
    )(service.image_analyze)
    
    mcp.tool(
        description="Edit or inpaint an image and write the result to the output reference.",
    )(service.image_edit)
    
    # ... additional tools
    
    return mcp
```

### Advantages of Functional Registration

1. **Dependency Injection**: Clean injection of settings, client, and logger through constructor parameters
2. **Testability**: Easy to create mock services and test different configurations
3. **Flexibility**: Can instantiate multiple server instances with different configurations
4. **Encapsulation**: The `ToolService` class encapsulates shared logic and state
5. **No Global State**: Avoids global mcp instance and related coupling issues

### Disadvantages of Functional Registration

1. **More Verbose**: Requires explicit registration calls for each tool
2. **Separation**: Tool definitions are separated from their registration (though still co-located)
3. **Additional Boilerplate**: Need to create service instance before registration

## Alternative Approach: Decorator-Based Registration

The more common FastMCP pattern would be decorator-based registration:

```python
mcp = FastMCP("multimodal-mcp")

@mcp.tool(description="Generate an image from a prompt and write it to the output reference.")
def image_generate(
    prompt: str,
    output_ref: str,
    size: Optional[str] = None,
    background: Optional[str] = None,
    quality: Optional[str] = None,
    format: Optional[str] = None,
    overwrite: bool = False,
    seed: Optional[int] = None,
    safety: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    output_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate an image from a prompt and write it to the output reference."""
    request_id = str(uuid.uuid4())
    warnings: List[str] = []
    # ... implementation
    return result

@mcp.tool(description="Analyze an image and return text or schema-validated JSON.")
def image_analyze(
    image_ref: str,
    instruction: str,
    response_format: str = "text",
    json_schema: Optional[Dict[str, Any]] = None,
    max_output_tokens: Optional[int] = None,
    detail: Optional[str] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze an image and return text or schema-validated JSON."""
    # ... implementation
    return result

# ... additional tools
```

### Advantages of Decorator Approach

1. **Concise and Declarative**: More compact syntax with clear intent
2. **Convention**: Follows common Python patterns (Flask, FastAPI, etc.)
3. **Co-location**: Registration is directly at function definition
4. **Less Boilerplate**: No separate registration step needed

### Disadvantages of Decorator Approach

1. **Dependency Management**: Difficult to inject dependencies cleanly
   - Would need global variables or closures for settings, client, logger
   - Makes testing harder (can't easily mock dependencies)
2. **Global State**: The `mcp` instance becomes global
3. **Limited Flexibility**: Harder to create multiple server instances with different configs
4. **Code Duplication**: Shared logic (error handling, logging) must be duplicated or extracted awkwardly

## Comparison: Handling Dependencies

### Functional Registration (Current)
```python
class ToolService:
    def __init__(self, settings: Settings, client: OpenAIClientProtocol, logger: logging.Logger):
        self._settings = settings
        self._client = client
        self._logger = logger
    
    def image_generate(self, ...):
        # Direct access to self._settings, self._client, self._logger
        request_id = self._new_request_id()
        self._log_info("image_generate start", request_id)
        result = self._client.generate_image(...)
```

### Decorator Approach (Alternative)
```python
# Option 1: Global variables (not ideal)
settings = load_settings()
client = OpenAIClient(settings)
logger = setup_logging(settings.log_level)

@mcp.tool(...)
def image_generate(...):
    # Access global settings, client, logger
    result = client.generate_image(...)

# Option 2: Closure (better but still awkward)
def create_tools(settings, client, logger):
    @mcp.tool(...)
    def image_generate(...):
        # Access via closure
        result = client.generate_image(...)
    
    return [image_generate, ...]  # But then what?
```

## Design Decision Rationale

**The functional registration approach is the correct choice for this codebase** because:

1. **Production Quality**: This is a production-grade server requiring clean dependency management
2. **Testability**: Unit tests can inject mock clients and settings easily
3. **Maintainability**: The `ToolService` class provides a clear boundary for shared logic
4. **Configurability**: Different server instances can be created with different configurations
5. **No Compromises**: Avoids compromising code quality for syntactic sugar

## When to Use Each Approach

### Use Functional Registration When:
- You need dependency injection
- Testing is important
- You have shared state or logic across tools
- Multiple configurations or instances are needed
- Building production systems

### Use Decorator Approach When:
- Building simple prototypes or demos
- Tools are completely independent
- No shared dependencies or state
- Convention over configuration is prioritized
- Quick iteration is more important than testability

## Conclusion

While the decorator pattern is more common in framework examples and tutorials, the functional registration pattern used in this codebase is **more appropriate for production systems**. It prioritizes:

- Clean architecture
- Testability
- Maintainability
- Flexibility

over

- Syntactic brevity
- Following common conventions

This is a conscious design choice that reflects software engineering best practices for production services.

---

## Server Architecture 

The Multimodal MCP Server is built using a layered architecture that separates concerns between protocol handling, business logic, external API integration, and I/O operations. This section provides visual representations and explanations of the system architecture.

### Context Diagram (Level 1)

```mermaid
C4Context
    title System Context - Multimodal MCP Server

    Person(user, "MCP Client", "Claude Desktop or other MCP-compatible client")
    System(mcpServer, "Multimodal MCP Server", "Provides multimodal AI capabilities via MCP protocol")
    System_Ext(openai, "OpenAI API", "GPT-4V, DALL-E, Whisper, TTS")
    System_Ext(filesystem, "File System", "Local files and directories")
    System_Ext(http, "HTTP Resources", "Remote images and audio files")

    Rel(user, mcpServer, "Sends tool requests", "MCP Protocol")
    Rel(mcpServer, openai, "API calls", "HTTPS/REST")
    Rel(mcpServer, filesystem, "Reads/writes files", "File I/O")
    Rel(mcpServer, http, "Downloads resources", "HTTPS")
```

### Container Diagram (Level 2)

```mermaid
C4Container
    title Container Diagram - Multimodal MCP Server

    Person(user, "MCP Client", "Claude Desktop")
    
    Container_Boundary(server, "Multimodal MCP Server") {
        Container(fastmcp, "FastMCP", "Python/FastMCP", "MCP protocol handler")
        Container(toolservice, "ToolService", "Python", "Business logic and orchestration")
        Container(openaiClient, "OpenAIClient", "Python", "OpenAI API integration")
        Container(fileref, "FileRef Module", "Python", "File I/O and URL handling")
        Container(config, "Config Module", "Python", "Settings and configuration")
    }
    
    System_Ext(openai, "OpenAI API", "External AI services")
    System_Ext(storage, "Storage", "Files, URLs")

    Rel(user, fastmcp, "Tool calls", "MCP/JSON-RPC")
    Rel(fastmcp, toolservice, "Invokes methods")
    Rel(toolservice, openaiClient, "API requests")
    Rel(toolservice, fileref, "Read/write data")
    Rel(toolservice, config, "Gets settings")
    Rel(openaiClient, openai, "HTTPS/REST")
    Rel(fileref, storage, "I/O operations")
```

### Component Diagram (Level 3)

```mermaid
C4Component
    title Component Diagram - ToolService & Dependencies

    Container_Boundary(toolservice, "ToolService") {
        Component(imageTools, "Image Tools", "Python methods", "image_generate, image_analyze, image_edit, image_extract, image_to_spec")
        Component(audioTools, "Audio Tools", "Python methods", "audio_transcribe, audio_analyze, audio_transform, audio_tts")
        Component(chainTool, "Chain Tool", "Python method", "multimodal_chain - orchestrates multi-step workflows")
        Component(errorHandler, "Error Handler", "Python methods", "_error_result, logging")
    }
    
    Component(openaiClientComp, "OpenAIClient", "Python class", "Wraps OpenAI SDK with retry logic")
    Component(filerefComp, "FileRef", "Python module", "read_input, write_output_bytes, write_output_text")
    Component(schemas, "Schemas", "Pydantic models", "Request/response validation")
    Component(errors, "Errors", "Python module", "MCPError, error codes")
    
    Rel(imageTools, openaiClientComp, "Uses")
    Rel(audioTools, openaiClientComp, "Uses")
    Rel(imageTools, filerefComp, "Read/write")
    Rel(audioTools, filerefComp, "Read/write")
    Rel(chainTool, imageTools, "Orchestrates")
    Rel(chainTool, audioTools, "Orchestrates")
    Rel(imageTools, schemas, "Validates with")
    Rel(audioTools, schemas, "Validates with")
    Rel(imageTools, errorHandler, "Reports errors")
    Rel(audioTools, errorHandler, "Reports errors")
```

## Class Diagram

```mermaid
classDiagram
    class FastMCP {
        +tool(description) decorator
        +run()
    }
    
    class ToolService {
        -Settings _settings
        -OpenAIClientProtocol _client
        -Logger _logger
        +image_generate() Dict
        +image_analyze() Dict
        +image_edit() Dict
        +image_extract() Dict
        +image_to_spec() Dict
        +audio_transcribe() Dict
        +audio_analyze() Dict
        +audio_transform() Dict
        +audio_tts() Dict
        +multimodal_chain() Dict
        -_new_request_id() str
        -_log_info()
        -_error_result() Dict
        -_chain_tool_handlers() Dict
    }
    
    class OpenAIClientProtocol {
        <<interface>>
        +analyze_image() ImageAnalysisResult
        +generate_image() ImageGenerationResult
        +edit_image() ImageEditResult
        +extract_image() ImageExtractResult
        +image_to_spec() ImageSpecResult
        +transcribe_audio() TranscriptionResult
        +analyze_audio() AudioAnalysisResult
        +transform_audio() AudioTransformResult
        +text_to_speech() SpeechResult
    }
    
    class OpenAIClient {
        -OpenAI _client
        -Settings _settings
        +analyze_image() ImageAnalysisResult
        +generate_image() ImageGenerationResult
        +edit_image() ImageEditResult
        +extract_image() ImageExtractResult
        +image_to_spec() ImageSpecResult
        +transcribe_audio() TranscriptionResult
        +analyze_audio() AudioAnalysisResult
        +transform_audio() AudioTransformResult
        +text_to_speech() SpeechResult
        -_transcribe() TranscriptionResult
        -_analyze_via_transcription() AudioAnalysisResult
        -_transform_via_transcription_and_tts() AudioTransformResult
    }
    
    class Settings {
        +str openai_api_key
        +str openai_base_url
        +str openai_model_vision
        +str openai_model_image
        +str openai_model_stt
        +str openai_model_tts
        +bool enable_remote_urls
        +bool enable_presigned_uploads
        +int max_input_bytes
        +int max_output_bytes
        +str log_level
        +Path temp_dir
    }
    
    class InputData {
        +bytes data
        +str mime_type
        +int size
        +str source
    }
    
    class OutputInfo {
        +str kind
        +str path_or_url
        +str mime_type
        +int bytes_written
        +str sha256
    }
    
    class ToolResult {
        +bool ok
        +List~OutputInfo~ outputs
        +Dict metadata
        +List~str~ warnings
        +ErrorInfo error
    }
    
    class ErrorInfo {
        +str code
        +str message
    }
    
    class MCPError {
        +str code
        +str message
        +Exception cause
    }
    
    FastMCP --> ToolService : registers tools from
    ToolService --> OpenAIClientProtocol : depends on
    OpenAIClient ..|> OpenAIClientProtocol : implements
    ToolService --> Settings : uses
    OpenAIClient --> Settings : uses
    ToolService --> InputData : reads via fileref
    ToolService --> OutputInfo : creates
    ToolService --> ToolResult : returns
    ToolService --> ErrorInfo : creates on error
    ToolService --> MCPError : handles
    ToolResult --> OutputInfo : contains
    ToolResult --> ErrorInfo : contains
```

## Architecture Layers

### Layer 1: Protocol Layer (FastMCP)

**Responsibility**: Handle MCP protocol, JSON-RPC communication, tool registration

```mermaid
flowchart LR
    Client[MCP Client] -->|JSON-RPC| FastMCP[FastMCP Framework]
    FastMCP -->|Method Call| ToolService[ToolService Methods]
    ToolService -->|JSON Response| FastMCP
    FastMCP -->|JSON-RPC| Client
```

**Key Characteristics**:
- Handles protocol serialization/deserialization
- Manages tool discovery and invocation
- Translates between MCP protocol and Python method calls

### Layer 2: Service Layer (ToolService)

**Responsibility**: Business logic, validation, orchestration, error handling

```mermaid
flowchart TD
    Tool[Tool Method] --> Validate[Validate Input]
    Validate --> ReadInput[Read Input Files/URLs]
    ReadInput --> CallAPI[Call OpenAI API]
    CallAPI --> ValidateOutput[Validate API Response]
    ValidateOutput --> WriteOutput[Write Output Files]
    WriteOutput --> BuildResult[Build ToolResult]
    BuildResult --> Return[Return Dict]
    
    Validate -->|Error| ErrorHandler[Error Handler]
    ReadInput -->|Error| ErrorHandler
    CallAPI -->|Error| ErrorHandler
    ValidateOutput -->|Error| ErrorHandler
    WriteOutput -->|Error| ErrorHandler
    ErrorHandler --> Return
```

**Key Characteristics**:
- Implements all tool methods (image_generate, audio_transcribe, etc.)
- Validates arguments using Pydantic schemas
- Coordinates between fileref and OpenAI client
- Consistent error handling and logging
- Returns standardized ToolResult dictionaries

### Layer 3: Integration Layer (OpenAIClient)

**Responsibility**: OpenAI API integration, retry logic, response transformation

```mermaid
flowchart TD
    Request[API Request] --> Retry{Retry Logic}
    Retry -->|Attempt| APICall[OpenAI SDK Call]
    APICall -->|Success| Transform[Transform Response]
    APICall -->|Transient Error| Retry
    APICall -->|Permanent Error| Error[Raise MCPError]
    Transform --> Result[Return Result Object]
```

**Key Characteristics**:
- Wraps OpenAI Python SDK
- Implements retry logic with exponential backoff
- Handles rate limits and transient errors
- Transforms OpenAI responses to internal result types
- Supports model overrides and configuration

### Layer 4: I/O Layer (FileRef Module)

**Responsibility**: File and URL I/O, validation, security checks

```mermaid
flowchart TD
    Ref[File Reference] --> IsURL{Is URL?}
    IsURL -->|Yes| CheckRemote{Remote Enabled?}
    IsURL -->|No| CheckLocal{Local Path Valid?}
    CheckRemote -->|Yes| Download[Download via HTTPS]
    CheckRemote -->|No| Error1[Error: Remote Disabled]
    CheckLocal -->|Yes| ReadFile[Read Local File]
    CheckLocal -->|No| Error2[Error: Invalid Path]
    Download --> Validate[Validate Size/Type]
    ReadFile --> Validate
    Validate --> Return[Return InputData]
    Validate -->|Too Large| Error3[Error: Too Large]
```

**Key Characteristics**:
- Handles both local files and remote URLs
- Enforces size limits and security policies
- Computes SHA256 hashes for integrity
- Supports presigned URL uploads
- Provides consistent InputData abstraction

## Data Flow: Image Generation Example

```mermaid
sequenceDiagram
    participant Client as MCP Client
    participant FastMCP
    participant Service as ToolService
    participant OpenAI as OpenAIClient
    participant API as OpenAI API
    participant FileRef
    participant FS as File System

    Client->>FastMCP: image_generate(prompt, output_ref)
    FastMCP->>Service: image_generate(...)
    
    Service->>Service: Validate arguments
    Service->>Service: Generate request_id
    Service->>Service: Log start
    
    Service->>OpenAI: generate_image(prompt, ...)
    OpenAI->>API: POST /v1/images/generations
    API-->>OpenAI: Image bytes
    OpenAI-->>Service: ImageGenerationResult
    
    Service->>FileRef: write_output_bytes(output_ref, data)
    FileRef->>FileRef: Validate output path
    FileRef->>FS: Write file
    FileRef->>FileRef: Compute SHA256
    FileRef-->>Service: OutputInfo
    
    Service->>Service: Build ToolResult
    Service->>Service: Log end
    Service-->>FastMCP: Dict (ToolResult)
    FastMCP-->>Client: JSON Response
```

## Data Flow: Multimodal Chain Example

```mermaid
sequenceDiagram
    participant Client as MCP Client
    participant Service as ToolService
    participant OpenAI as OpenAIClient
    participant FileRef

    Client->>Service: multimodal_chain(steps=[...])
    
    Note over Service: Step 1: Generate Image
    Service->>Service: image_generate(prompt="sunset")
    Service->>OpenAI: generate_image(...)
    OpenAI-->>Service: image bytes
    Service->>FileRef: write("/tmp/sunset.png")
    Service->>Service: Store result as "step1"
    
    Note over Service: Step 2: Analyze Image
    Service->>Service: image_analyze(image_ref="${step1.outputs[0].path}")
    Service->>Service: Resolve reference to "/tmp/sunset.png"
    Service->>FileRef: read_input("/tmp/sunset.png")
    FileRef-->>Service: image bytes
    Service->>OpenAI: analyze_image(...)
    OpenAI-->>Service: analysis text
    Service->>Service: Store result as "step2"
    
    Note over Service: Step 3: Generate Speech
    Service->>Service: audio_tts(text="${step2.metadata.text}")
    Service->>Service: Resolve reference to analysis text
    Service->>OpenAI: text_to_speech(...)
    OpenAI-->>Service: audio bytes
    Service->>FileRef: write("/tmp/description.mp3")
    
    Service->>Service: Build chain result
    Service-->>Client: ToolResult with all outputs
```

## Error Handling Architecture

```mermaid
flowchart TD
    Operation[Tool Operation] --> Success{Success?}
    Success -->|Yes| BuildSuccess[Build ToolResult ok=True]
    Success -->|No| ExceptionType{Exception Type}
    
    ExceptionType -->|ValidationError| InvalidArg[ErrorInfo: INVALID_ARGUMENT]
    ExceptionType -->|MCPError| MCPErr[ErrorInfo from MCPError]
    ExceptionType -->|Other| InternalErr[ErrorInfo: INTERNAL_ERROR]
    
    InvalidArg --> Log1[Log Error]
    MCPErr --> Log2[Log Error with Code]
    InternalErr --> Log3[Log Error with Stack Trace]
    
    Log1 --> BuildError[Build ToolResult ok=False]
    Log2 --> BuildError
    Log3 --> BuildError
    BuildSuccess --> Return[Return Dict]
    BuildError --> Return
    
    style BuildError fill:#f88
    style BuildSuccess fill:#8f8
```

**Error Codes**:
- `INVALID_ARGUMENT`: Validation failures, bad parameters
- `INPUT_NOT_FOUND`: Missing input files
- `INPUT_TOO_LARGE`: Input exceeds size limits
- `OUTPUT_EXISTS`: Output file exists, overwrite not allowed
- `UNSUPPORTED_FORMAT`: Unsupported file format
- `OPENAI_ERROR`: OpenAI API errors
- `SCHEMA_VALIDATION_FAILED`: JSON schema validation failures
- `CHAIN_STEP_FAILED`: Chain step execution failures
- `INTERNAL_ERROR`: Unexpected errors

## Configuration and Dependency Injection

```mermaid
flowchart TD
    ENV[Environment Variables] --> LoadSettings[load_settings]
    LoadSettings --> Settings[Settings Object]
    
    Settings --> Client[OpenAIClient]
    Settings --> Logger[Logger]
    Settings --> Service[ToolService]
    Client --> Service
    Logger --> Service
    
    Service --> FastMCP[FastMCP Instance]
    
    style Settings fill:#9cf
    style Service fill:#fc9
    style FastMCP fill:#9f9
```

**Settings Sources** (in order of precedence):
1. Environment variables
2. `.env` file
3. Default values

**Injected Dependencies**:
- `Settings`: Configuration object
- `OpenAIClientProtocol`: AI service integration (mockable)
- `Logger`: Structured logging

This architecture enables:
- **Testability**: Mock any dependency
- **Flexibility**: Swap implementations (e.g., different AI providers)
- **Configuration**: Environment-based settings without code changes
- **Separation of Concerns**: Clear boundaries between layers

## Key Design Patterns

### 1. Dependency Injection
All dependencies flow from `build_server()` down through constructor injection, avoiding global state.

### 2. Protocol/Interface Segregation
`OpenAIClientProtocol` defines the interface, allowing mock implementations for testing.

### 3. Error Translation
Low-level exceptions (httpx, OpenAI SDK) are translated to domain-specific `MCPError` instances with meaningful error codes.

### 4. Consistent Result Format
All tools return `ToolResult` dictionaries with consistent structure (ok, outputs, metadata, warnings, error).

### 5. Reference Resolution
The chain tool uses a `$ref` syntax to reference outputs from previous steps, enabling complex workflows.

### 6. Retry with Exponential Backoff
Transient API failures are automatically retried with exponential backoff to improve reliability.

### 7. Security-First I/O
File and URL operations enforce configurable security policies (size limits, path validation, remote access control).
