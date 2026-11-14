abstract type AbstractGoogleProvider end

"""
    RetryOptions(; attempts=5, initial_delay=1.0, max_delay=60.0, exp_base=2.0, jitter=1.0, http_status_codes=[408, 429, 500, 502, 503, 504])

Retry policy for failed API requests. Pass it to [`GoogleProvider`](@ref) as `retry_options`;
requests are not retried without it.

A request is tried at most `attempts` times, and only retried when the response status is in
`http_status_codes`. Before retry `n` the client waits
`initial_delay * exp_base^(n - 1) + rand() * jitter` seconds, or, when the server says how
long to wait (a `Retry-After` header or a `RetryInfo` error detail), that long; either way
at most `max_delay`. Streaming requests are only retried before the first chunk.
"""
Base.@kwdef struct RetryOptions
    attempts::Int = 5
    initial_delay::Float64 = 1.0
    max_delay::Float64 = 60.0
    exp_base::Float64 = 2.0
    jitter::Float64 = 1.0
    http_status_codes::Vector{Int} = [408, 429, 500, 502, 503, 504]
end

"""
    GoogleProvider(; api_key="", base_url="https://generativelanguage.googleapis.com", api_version="v1beta", retry_options=nothing, http_options=(;))

A configuration object used to set up and authenticate requests to the Google Generative Language API.

# Fields
- `api_key::String`: Your Google API key. If not provided, the constructor will automatically check for `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variables (with `GOOGLE_API_KEY` taking precedence if both are set).
- `base_url::String`: The base URL for the Google Generative Language API. The default is set to `"https://generativelanguage.googleapis.com"`.
- `api_version::String`: The version of the API you wish to access. The default is set to `"v1beta"`.
- `retry_options::Union{Nothing,RetryOptions}`: Retry policy for failed requests; see [`RetryOptions`](@ref). `nothing` disables retries.
- `http_options::NamedTuple`: Default keyword arguments for every `HTTP.request` call, e.g. `(readtimeout=60,)`. Per-call `http_options`/`http_kwargs` override these.
"""
struct GoogleProvider <: AbstractGoogleProvider
    api_key::String
    base_url::String
    api_version::String
    retry_options::Union{Nothing,RetryOptions}
    http_options::NamedTuple
end

function GoogleProvider(api_key::String, base_url::String, api_version::String)
    return GoogleProvider(api_key, base_url, api_version, nothing, (;))
end

function GoogleProvider(;
    api_key::String=get(ENV, "GOOGLE_API_KEY", get(ENV, "GEMINI_API_KEY", "")),
    base_url::String="https://generativelanguage.googleapis.com",
    api_version::String="v1beta",
    retry_options::Union{Nothing,RetryOptions}=nothing,
    http_options::NamedTuple=(;),
)
    if isempty(api_key)
        error(
            "API key not provided and not found in GOOGLE_API_KEY or GEMINI_API_KEY environment variables.",
        )
    end

    return GoogleProvider(api_key, base_url, api_version, retry_options, http_options)
end

function Base.show(io::IO, ::MIME"text/plain", provider::GoogleProvider)
    api_key = provider.api_key
    redacted_key = length(api_key) > 10 ? "...$(api_key[end-2:end])" : "<hidden>"
    retries = provider.retry_options
    println(io, "GoogleProvider:")
    println(io, "  api_key:     \"$(redacted_key)\"")
    println(io, "  base_url:    \"$(provider.base_url)\"")
    println(io, "  api_version: \"$(provider.api_version)\"")
    return print(
        io, "  retries:     ", retries === nothing ? "off" : "$(retries.attempts) attempts"
    )
end

"""
    GeminiAPIError <: Exception

Thrown when the API answers with an error status.

# Fields
- `code::Int`: HTTP status code, e.g. `429`.
- `status::String`: Google RPC status, e.g. `"RESOURCE_EXHAUSTED"`; empty if the body had none.
- `message::String`: The error message from the API (the raw body if it was not JSON).
- `details`: The error's `details` array, or `nothing`.
- `retry_delay::Union{Nothing,Float64}`: Seconds the server asked the client to wait before retrying, if it said.
"""
struct GeminiAPIError <: Exception
    code::Int
    status::String
    message::String
    details::Any
    retry_delay::Union{Nothing,Float64}
end

function GeminiAPIError(code::Integer, body::AbstractString, retry_delay=nothing)
    err = try
        parsed = JSON3.read(body)
        parsed isa JSON3.Array && !isempty(parsed) && (parsed = first(parsed))
        parsed isa JSON3.Object ? get(parsed, :error, parsed) : nothing
    catch
        nothing
    end
    err isa JSON3.Object || return GeminiAPIError(code, "", body, nothing, retry_delay)
    details = get(err, :details, nothing)
    api_code = get(err, :code, code)
    return GeminiAPIError(
        api_code isa Integer ? Int(api_code) : Int(code),
        String(get(err, :status, api_code isa AbstractString ? api_code : "")),
        String(get(err, :message, body)),
        details,
        something(retry_delay, _retry_info_delay(details), Some(nothing)),
    )
end

function GeminiAPIError(response::HTTP.Response)
    retry_after = tryparse(Float64, HTTP.header(response, "Retry-After", ""))
    return GeminiAPIError(response.status, String(copy(response.body)), retry_after)
end

function _retry_info_delay(details)
    details isa AbstractVector || return nothing
    for detail in details
        detail isa AbstractDict || continue
        endswith(string(get(detail, Symbol("@type"), "")), "RetryInfo") || continue
        m = match(r"^(\d+(?:\.\d+)?)s$", string(get(detail, :retryDelay, "")))
        m === nothing || return parse(Float64, m[1])
    end
    return nothing
end

function Base.showerror(io::IO, e::GeminiAPIError)
    print(io, "GeminiAPIError: ", e.code)
    isempty(e.status) || print(io, " ", e.status)
    return print(io, ": ", e.message)
end

const VALID_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
]

const VALID_THRESHOLDS = [
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "OFF",
]

"""
    SafetySetting

# Fields
- `category::String`: The type of harmful content to filter. Must be one of:
  - `"HARM_CATEGORY_HARASSMENT"`
  - `"HARM_CATEGORY_HATE_SPEECH"`
  - `"HARM_CATEGORY_SEXUALLY_EXPLICIT"`
  - `"HARM_CATEGORY_DANGEROUS_CONTENT"`
  - `"HARM_CATEGORY_CIVIC_INTEGRITY"`
- `threshold::String`: The sensitivity level for blocking content. Must be one of:
  - `"BLOCK_NONE"`: Never block.
  - `"BLOCK_ONLY_HIGH"`: Block only content with a high likelihood of harm.
  - `"BLOCK_MEDIUM_AND_ABOVE"`: Block when medium or high probability of unsafe content
  - `"BLOCK_LOW_AND_ABOVE"`: Block when low, medium or high probability of unsafe content
  - `"HARM_BLOCK_THRESHOLD_UNSPECIFIED"`: Threshold is unspecified, block using default threshold
  - `"OFF"`: Turn the safety filter off.
"""
Base.@kwdef struct SafetySetting
    category::String
    threshold::String
    function SafetySetting(category::String, threshold::String)
        if !(category in VALID_CATEGORIES)
            throw(
                ArgumentError(
                    "Invalid category: '$category'. Must be one of: $VALID_CATEGORIES"
                ),
            )
        elseif !(threshold in VALID_THRESHOLDS)
            throw(
                ArgumentError(
                    "Invalid threshold: '$threshold'. Must be one of: $VALID_THRESHOLDS"
                ),
            )
        end
        return new(category, threshold)
    end
end

"""
    ToolType

The tools the Gemini API provides, as identified by [`is_native_tool`](@ref).

# Values
- `GOOGLE_SEARCH`: Grounding with Google Search.
- `CODE_EXECUTION`: The model writes and runs Python code.
- `FUNCTION_CALLING`: User-defined function declarations.
- `URL_CONTEXT`: The model reads URLs given in the prompt.
- `GOOGLE_MAPS`: Grounding with Google Maps.
- `FILE_SEARCH`: Retrieval from file search stores.
- `COMPUTER_USE`: The model operates a browser or device through UI actions.
- `MCP_SERVERS`: Remote MCP servers called by the API.
"""
@enum ToolType begin
    GOOGLE_SEARCH
    CODE_EXECUTION
    FUNCTION_CALLING
    URL_CONTEXT
    GOOGLE_MAPS
    FILE_SEARCH
    COMPUTER_USE
    MCP_SERVERS
end

const TOOL_KEYS = (
    "googleSearch" => GOOGLE_SEARCH,
    "codeExecution" => CODE_EXECUTION,
    "functionDeclarations" => FUNCTION_CALLING,
    "urlContext" => URL_CONTEXT,
    "googleMaps" => GOOGLE_MAPS,
    "fileSearch" => FILE_SEARCH,
    "computerUse" => COMPUTER_USE,
    "mcpServers" => MCP_SERVERS,
)

"""
    is_native_tool(tool::AbstractDict) -> Tuple{Bool, Union{ToolType, Nothing}}

Check whether a tool dictionary is one of the API's tools and return its [`ToolType`](@ref).
"""
function is_native_tool(tool::AbstractDict)
    for (key, type) in TOOL_KEYS
        _key(tool, Symbol(key)) === nothing || return true, type
    end
    return false, nothing
end

function _tool(key; fields...)
    return Dict{String,Any}(key => _compact((string(k) => v for (k, v) in fields)...))
end

"""
    google_search_tool()

Tool that grounds answers with Google Search.
"""
google_search_tool() = _tool("googleSearch")

"""
    code_execution_tool()

Tool that lets the model write and run Python code.
"""
code_execution_tool() = _tool("codeExecution")

"""
    url_context_tool()

Tool that lets the model read the URLs given in the prompt.
"""
url_context_tool() = _tool("urlContext")

"""
    google_maps_tool(; enable_widget=nothing)

Tool that grounds answers with Google Maps. `enable_widget=true` returns a context token for
rendering a Maps widget.
"""
google_maps_tool(; enable_widget=nothing) = _tool("googleMaps"; enableWidget=enable_widget)

"""
    file_search_tool(store_names; top_k=nothing, metadata_filter=nothing)

Tool that retrieves from the given file search stores (e.g. `"fileSearchStores/abc"`).
"""
function file_search_tool(store_names; top_k=nothing, metadata_filter=nothing)
    return _tool(
        "fileSearch";
        fileSearchStoreNames=collect(String, store_names),
        topK=top_k,
        metadataFilter=metadata_filter,
    )
end

"""
    computer_use_tool(; environment="ENVIRONMENT_BROWSER", excluded_predefined_functions=nothing)

Tool that lets the model operate a browser (or `"ENVIRONMENT_MOBILE"` device) through UI actions.
"""
function computer_use_tool(;
    environment="ENVIRONMENT_BROWSER", excluded_predefined_functions=nothing
)
    return _tool(
        "computerUse";
        environment,
        excludedPredefinedFunctions=excluded_predefined_functions,
    )
end

"""
    FunctionParameter

Represents a parameter for a function declaration.

# Fields
- `type::String`: The type of the parameter (e.g., "object", "string", "array", etc.).
- `description::Union{String, Nothing}`: Optional description of the parameter.
- `properties::Union{Dict{String, Any}, Nothing}`: For "object" type, defines the properties.
- `items::Union{Dict{String, Any}, Nothing}`: For "array" type, defines the array items.
- `required::Union{Vector{String}, Nothing}`: For "object" type, lists required property names.
"""
Base.@kwdef struct FunctionParameter
    type::String
    description::Union{String,Nothing} = nothing
    properties::Union{Dict{String,Any},Nothing} = nothing
    items::Union{Dict{String,Any},Nothing} = nothing
    required::Union{Vector{String},Nothing} = nothing
end

"""
    FunctionDeclaration

Represents a function that the model can call.

# Fields
- `name::String`: The name of the function.
- `description::Union{String, Nothing}`: Optional description of the function.
- `parameters::FunctionParameter`: The parameters schema for the function.
"""
Base.@kwdef struct FunctionDeclaration
    name::String
    description::Union{String,Nothing} = nothing
    parameters::FunctionParameter
end

"""
    FunctionCall

Represents a function call from the model.

# Fields
- `name::String`: The name of the function to call.
- `args::Dict{String, Any}`: The arguments for the function call.
"""
Base.@kwdef struct FunctionCall
    name::String
    args::Dict{String,Any}
end

"""
    FunctionCallingConfig

Controls how the model uses function declarations.

# Fields
- `mode`: "AUTO" (default), "ANY", or "NONE"
- `allowed_function_names`: Optional list of allowed functions when mode is "ANY"
"""
Base.@kwdef struct FunctionCallingConfig
    mode::String = "AUTO"
    allowed_function_names::Union{Nothing,Vector{String}} = nothing

    function FunctionCallingConfig(
        mode::String, allowed_function_names::Union{Nothing,Vector{String}}=nothing
    )
        if !(mode in ["AUTO", "ANY", "NONE"])
            throw(ArgumentError("mode must be one of: AUTO, ANY, NONE"))
        end
        return new(mode, allowed_function_names)
    end
end

"""
    ToolConfig

Configuration for tools behavior.

# Fields
- `function_calling_config`: Configuration for function calling
"""
Base.@kwdef struct ToolConfig
    function_calling_config::Union{Nothing,FunctionCallingConfig} = nothing
end

const THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH")

"""
    ThinkingConfig(; include_thoughts=nothing, thinking_budget=nothing, thinking_level=nothing)

Controls the model's internal thinking. Unset fields are left to the model's default.

See https://ai.google.dev/gemini-api/docs/thinking.

# Fields
- `include_thoughts::Union{Nothing,Bool}`: Return thought summaries in the response.
- `thinking_budget::Union{Nothing,Int}`: Thinking budget in tokens (`-1` for dynamic, `0` to turn thinking off where the model allows it). Used by Gemini 2.5 models.
- `thinking_level::Union{Nothing,String}`: One of `"minimal"`, `"low"`, `"medium"`, `"high"` or `"xhigh"` (case-insensitive). Used by Gemini 3 models.

Set at most one of `thinking_budget` and `thinking_level`.
"""
Base.@kwdef struct ThinkingConfig
    include_thoughts::Union{Nothing,Bool} = nothing
    thinking_budget::Union{Nothing,Int} = nothing
    thinking_level::Union{Nothing,String} = nothing

    function ThinkingConfig(include_thoughts, thinking_budget, thinking_level)
        if thinking_level !== nothing
            thinking_level = uppercase(thinking_level)
            thinking_level in THINKING_LEVELS || throw(
                ArgumentError(
                    "thinking_level must be one of: $(join(lowercase.(THINKING_LEVELS), ", "))",
                ),
            )
            thinking_budget === nothing ||
                throw(ArgumentError("set only one of thinking_budget and thinking_level"))
        end
        return new(include_thoughts, thinking_budget, thinking_level)
    end
end

"""
    GenerateContentConfig

Optional model configuration parameters.

# Fields
- `http_options=(;)`: Keyword arguments passed to `HTTP.request` for this call, overriding the provider's `http_options`. See https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.
- `system_instruction`: Instructions for the model: a string, a vector of strings and parts, or a content dict with `parts`.
- `temperature::Union{Nothing,Float64}`: Controls the degree of randomness in token selection.
- `top_p::Union{Nothing,Float64}`: Selects tokens from most to least probable until the sum of their probabilities equals this value.
- `top_k::Union{Nothing,Float64}`: Samples the `top_k` tokens with the highest probabilities at each step.
- `candidate_count::Union{Nothing,Int}`: Number of response variations to return.
- `max_output_tokens::Union{Nothing,Int}`: Maximum number of tokens that can be generated.
- `stop_sequences::Union{Nothing,Vector{String}}`: List of strings that tell the model to stop generating text.
- `response_logprobs::Union{Nothing,Bool}`: Whether to return the log probabilities of chosen tokens.
- `logprobs::Union{Nothing,Int}`: Number of top candidate tokens to return log probabilities for.
- `presence_penalty::Union{Nothing,Float64}`: Penalizes tokens that already appear, increasing diversity.
- `frequency_penalty::Union{Nothing,Float64}`: Penalizes tokens that appear repeatedly, increasing diversity.
- `seed::Union{Nothing,Int}`: Fixed seed for reproducibility; otherwise, a random number is used.
- `response_mime_type::Union{Nothing,String}`: Output response media type.
- `response_schema::Union{Nothing,AbstractDict}`: Schema (OpenAPI subset) that the generated candidate text must adhere to.
- `response_json_schema::Union{Nothing,AbstractDict}`: A JSON Schema the output must follow. Use instead of `response_schema`.
- `safety_settings::Union{Nothing,Vector{SafetySetting}}`: Safety settings to block unsafe content.
- `tools::Union{Nothing,Vector{Any}}`: Tools the model may use: tool dicts (see [`google_search_tool`](@ref), [`url_context_tool`](@ref), [`code_execution_tool`](@ref), [`google_maps_tool`](@ref), [`file_search_tool`](@ref), [`computer_use_tool`](@ref)), [`FunctionDeclaration`](@ref)s, or Julia functions (declared by name with no parameters).
- `function_declarations::Union{Nothing,Vector{FunctionDeclaration}}`: Declarations of functions that the model can call.
- `tool_config::Union{Nothing,ToolConfig} = nothing`: Associates model output to a specific function call.
- `labels::Union{Nothing,Dict{String,String}}`: User-defined metadata labels.
- `cached_content::Union{Nothing,String}`: Resource name of a context cache.
- `response_modalities::Union{Nothing,Vector{String}}`: Requested modalities of the response.
- `media_resolution::Union{Nothing,String}`: Specified media resolution.
- `speech_config::Union{Nothing,AbstractDict}`: Speech generation configuration.
- `thinking_config::Union{Nothing,ThinkingConfig}`: Thinking features configuration.
- `image_config::Union{Nothing,AbstractDict}`: Image generation options, e.g. `Dict(:aspectRatio => "16:9", :imageSize => "2K")`.
- `service_tier::Union{Nothing,String}`: Processing tier: `"standard"`, `"flex"` or `"priority"`.
"""
Base.@kwdef struct GenerateContentConfig
    http_options = (;)
    system_instruction::Union{Nothing,AbstractString,AbstractDict,AbstractVector} = nothing
    temperature::Union{Nothing,Float64} = nothing
    top_p::Union{Nothing,Float64} = nothing
    top_k::Union{Nothing,Float64} = nothing
    candidate_count::Union{Nothing,Int} = nothing
    max_output_tokens::Union{Nothing,Int} = nothing
    stop_sequences::Union{Nothing,Vector{String}} = nothing
    response_logprobs::Union{Nothing,Bool} = nothing
    logprobs::Union{Nothing,Int} = nothing
    presence_penalty::Union{Nothing,Float64} = nothing
    frequency_penalty::Union{Nothing,Float64} = nothing
    seed::Union{Nothing,Int} = nothing
    response_mime_type::Union{Nothing,String} = nothing
    response_schema::Union{Nothing,AbstractDict} = nothing
    response_json_schema::Union{Nothing,AbstractDict} = nothing
    safety_settings::Union{Nothing,Vector{SafetySetting}} = nothing
    tools::Union{Nothing,Vector{Any}} = nothing
    function_declarations::Union{Nothing,Vector{FunctionDeclaration}} = nothing
    tool_config::Union{Nothing,ToolConfig} = nothing
    labels::Union{Nothing,Dict{String,String}} = nothing
    cached_content::Union{Nothing,String} = nothing
    response_modalities::Union{Nothing,Vector{String}} = nothing
    media_resolution::Union{Nothing,String} = nothing
    speech_config::Union{Nothing,AbstractDict} = nothing
    thinking_config::Union{Nothing,ThinkingConfig} = nothing
    image_config::Union{Nothing,AbstractDict} = nothing
    service_tier::Union{Nothing,String} = nothing
end

"""
    to_api_function_declaration(func_decl::FunctionDeclaration) -> Dict{String, Any}

Converts a FunctionDeclaration to the API-compatible dictionary format.
"""
function to_api_function_declaration(func_decl::FunctionDeclaration)
    p = func_decl.parameters
    params = _compact(
        "type" => p.type,
        "description" => p.description,
        "properties" => p.properties,
        "items" => p.items,
        "required" => p.required,
    )
    return _compact(
        "name" => func_decl.name,
        "description" => func_decl.description,
        "parameters" => params,
    )
end

# `GenerateContentConfig` field => `generationConfig` API key.
const GENERATION_CONFIG_FIELDS = (
    :temperature => "temperature",
    :top_p => "topP",
    :top_k => "topK",
    :candidate_count => "candidateCount",
    :max_output_tokens => "maxOutputTokens",
    :stop_sequences => "stopSequences",
    :response_logprobs => "responseLogprobs",
    :logprobs => "logprobs",
    :presence_penalty => "presencePenalty",
    :frequency_penalty => "frequencyPenalty",
    :seed => "seed",
    :response_mime_type => "responseMimeType",
    :response_schema => "responseSchema",
    :response_json_schema => "responseJsonSchema",
    :response_modalities => "responseModalities",
    :media_resolution => "mediaResolution",
    :speech_config => "speechConfig",
    :thinking_config => "thinkingConfig",
    :image_config => "imageConfig",
)

_api_value(x) = x
function _api_value(c::ThinkingConfig)
    return _compact(
        "includeThoughts" => c.include_thoughts,
        "thinkingBudget" => c.thinking_budget,
        "thinkingLevel" => c.thinking_level,
    )
end

function _build_generation_config(config::GenerateContentConfig)
    generation_config = Dict{String,Any}()
    for (field, key) in GENERATION_CONFIG_FIELDS
        value = getfield(config, field)
        value === nothing || (generation_config[key] = _api_value(value))
    end
    return generation_config
end
