abstract type AbstractGoogleProvider end

"""
    GoogleProvider(; api_key::String="", base_url::String="https://generativelanguage.googleapis.com", api_version::String="v1beta")

A configuration object used to set up and authenticate requests to the Google Generative Language API.

# Fields
- `api_key::String`: Your Google API key. If not provided, the constructor will automatically check for `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variables (with `GOOGLE_API_KEY` taking precedence if both are set).
- `base_url::String`: The base URL for the Google Generative Language API. The default is set to `"https://generativelanguage.googleapis.com"`.
- `api_version::String`: The version of the API you wish to access. The default is set to `"v1beta"`.
"""
struct GoogleProvider <: AbstractGoogleProvider
    api_key::String
    base_url::String
    api_version::String
end

function GoogleProvider(;
    api_key::String=get(ENV, "GOOGLE_API_KEY", get(ENV, "GEMINI_API_KEY", "")),
    base_url::String="https://generativelanguage.googleapis.com",
    api_version::String="v1beta",
)
    if isempty(api_key)
        error(
            "API key not provided and not found in GOOGLE_API_KEY or GEMINI_API_KEY environment variables.",
        )
    end

    return GoogleProvider(api_key, base_url, api_version)
end

function Base.show(io::IO, ::MIME"text/plain", provider::GoogleProvider)
    api_key = provider.api_key
    redacted_key = if length(api_key) > 10
        "...$(api_key[end-2:end])"
    else
        "<hidden>"
    end

    println(io, "GoogleProvider:")
    println(io, "  api_key:     \"$(redacted_key)\"")
    println(io, "  base_url:    \"$(provider.base_url)\"")
    return print(io, "  api_version: \"$(provider.api_version)\"")
end

"""
    APIError <: Exception

An error returned by the Gemini API.

# Fields
- `status::Int`: The HTTP status code, e.g. `429`.
- `code::String`: The API status name, e.g. `"RESOURCE_EXHAUSTED"` (empty if not given).
- `message::String`: The error message from the API.
- `details::Any`: Extra error details from the API, or `nothing`.

```julia
try
    generate_content(model, prompt)
catch e
    e isa APIError && e.status == 429 && retry_later()
end
```
"""
struct APIError <: Exception
    status::Int
    code::String
    message::String
    details::Any
end

function Base.showerror(io::IO, e::APIError)
    print(io, "APIError: ", e.status)
    isempty(e.code) || print(io, " ", e.code)
    return print(io, ": ", e.message)
end

_body_string(body::AbstractString) = String(body)
_body_string(body::AbstractVector{UInt8}) = String(copy(body))

# Build an `APIError` from Google's `{"error": {"code", "message", "status", "details"}}`
# payload. Streaming endpoints wrap it in an array; anything else is kept as the message.
function _api_error(status::Integer, body)
    text = _body_string(body)
    parsed = try
        JSON3.read(text)
    catch
        nothing
    end
    parsed isa JSON3.Array && length(parsed) == 1 && (parsed = parsed[1])
    err = parsed isa JSON3.Object ? get(parsed, :error, nothing) : nothing
    err isa JSON3.Object ||
        return APIError(status, "", isempty(text) ? "HTTP $status" : text, nothing)
    return APIError(
        Int(something(get(err, :code, nothing), status)),
        String(something(get(err, :status, nothing), "")),
        String(something(get(err, :message, nothing), text)),
        get(err, :details, nothing),
    )
end
_api_error(response::HTTP.Response) = _api_error(response.status, response.body)

function _check_status(response::HTTP.Response)
    response.status >= 400 && throw(_api_error(response))
    return response
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
  - `"BLOCK_NONE"`: 	Block when high probability of unsafe content is detected.
  - `"BLOCK_ONLY_HIGH"`: Block only content with a high likelihood of harm.
  - `"BLOCK_MEDIUM_AND_ABOVE"`: Block when medium or high probability of unsafe content
  - `"BLOCK_LOW_AND_ABOVE"`: Block when low, medium or high probability of unsafe content
  - `"HARM_BLOCK_THRESHOLD_UNSPECIFIED"`: Threshold is unspecified, block using default threshold
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

Enum representing the types of native tools supported by the Gemini API.

# Values
- `GOOGLE_SEARCH`: Represents the Google Search tool functionality that allows the model to search for information.
- `CODE_EXECUTION`: Represents the Code Execution tool which allows the model to execute code snippets.
- `FUNCTION_CALLING`: Represents the Function Calling capability that enables the model to call user-defined functions.
"""
@enum ToolType begin
    GOOGLE_SEARCH
    CODE_EXECUTION
    FUNCTION_CALLING
end

"""
    is_native_tool(tool::Dict{Symbol, Any}) -> Tuple{Bool, Union{ToolType, Nothing}}

Check if a tool dictionary represents a native tool and identify its type.

# Arguments
- `tool::Dict{Symbol, Any}`: The tool dictionary to check

# Returns
- `Tuple{Bool, Union{ToolType, Nothing}}`: A tuple with a boolean indicating if it's a native tool
  and the type of the tool if it is native, or nothing if it's not.
"""
function is_native_tool(tool::Dict{Symbol,Any})
    if haskey(tool, :googleSearch)
        return true, GOOGLE_SEARCH
    elseif haskey(tool, :codeExecution)
        return true, CODE_EXECUTION
    elseif haskey(tool, :functionDeclarations)
        return true, FUNCTION_CALLING
    else
        return false, nothing
    end
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

"""
    ThinkingConfig

Configuration for thinking features in Gemini models.

The Gemini models use an internal "thinking process" during response generation.
This process contributes to their improved reasoning capabilities and helps them use multi-step 
planning to solve complex tasks.

For more information, see: https://ai.google.dev/gemini-api/docs/thinking#set-budget

# Fields
- `include_thoughts::Bool`: Indicates whether to include thoughts in the response. If true, thoughts are returned only if the model supports thought and thoughts are available.
- `thinking_budget::Int`: Indicates the thinking budget in tokens. This limits the amount of internal thinking the model can perform.
"""
Base.@kwdef struct ThinkingConfig
    include_thoughts::Bool = false
    thinking_budget::Int = -1
end

"""
    GenerateContentConfig

Optional model configuration parameters.

# Fields
- `http_options=(;)`: Used to override HTTP request options.
    - All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.
- `system_instruction::Union{Nothing,String,Vector{Dict{Symbol,Any}}}`: Instructions for the model.
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
- `response_schema::Union{Nothing,Dict{Symbol,Any}}`: Schema that the generated candidate text must adhere to.
- `routing_config::Union{Nothing,Dict{Symbol,Any}}`: Configuration for model router requests.
- `safety_settings::Union{Nothing,Vector{SafetySetting}}`: Safety settings to block unsafe content.
- `tools::Union{Nothing,Vector{Any}}`: Enables interaction with external systems.
    This can include native tools, function declarations, or Functions for automatic schema generation:
    - Native tools are defined directly, e.g., `Dict(:googleSearch => Dict())`, `Dict(:codeExecution => Dict())`.
    - Function declarations are included using `Dict(:functionDeclarations => [declarations])`.
    - Julia Functions can be passed directly for automatic schema generation.
    - Helper functions are available: `create_google_search_tool()`, `create_code_execution_tool()`, `create_function_tool()`.
- `function_declarations::Union{Nothing,Vector{FunctionDeclaration}}`: Declarations of functions that the model can call.
    For multi-tool scenarios, consider using the `tools` field instead.
- `tool_config::Union{Nothing,ToolConfig} = nothing`: Associates model output to a specific function call.
- `labels::Union{Nothing,Dict{String,String}}`: User-defined metadata labels.
- `cached_content::Union{Nothing,String}`: Resource name of a context cache.
- `response_modalities::Union{Nothing,Vector{String}}`: Requested modalities of the response.
- `media_resolution::Union{Nothing,String}`: Specified media resolution.
- `speech_config::Union{Nothing,Dict{Symbol,Any}}`: Speech generation configuration.
- `audio_timestamp::Union{Nothing,Bool}`: Whether to include audio timestamp in the request.
- `thinking_config::Union{Nothing,ThinkingConfig}`: Thinking features configuration.
"""
Base.@kwdef struct GenerateContentConfig
    http_options = (;)
    system_instruction::Union{Nothing,String} = nothing
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
    response_schema::Union{Nothing,Dict{Symbol,Any}} = nothing
    routing_config::Union{Nothing,Dict{Symbol,Any}} = nothing
    safety_settings::Union{Nothing,Vector{SafetySetting}} = nothing
    tools::Union{Nothing,Vector{Any}} = nothing
    function_declarations::Union{Nothing,Vector{FunctionDeclaration}} = nothing
    tool_config::Union{Nothing,ToolConfig} = nothing
    labels::Union{Nothing,Dict{String,String}} = nothing
    cached_content::Union{Nothing,String} = nothing
    response_modalities::Union{Nothing,Vector{String}} = nothing
    media_resolution::Union{Nothing,String} = nothing
    speech_config::Union{Nothing,Dict{Symbol,Any}} = nothing
    audio_timestamp::Union{Nothing,Bool} = nothing
    thinking_config::Union{Nothing,ThinkingConfig} = nothing
end

# The API key goes in a header, never the URL, so it stays out of logs and errors.
function _headers(provider::AbstractGoogleProvider)
    return Dict("Content-Type" => "application/json", "x-goog-api-key" => provider.api_key)
end

function _request(
    provider::AbstractGoogleProvider,
    endpoint::String,
    method::Symbol,
    body::Dict;
    http_kwargs...,
)
    if isempty(provider.api_key)
        throw(ArgumentError("api_key cannot be empty"))
    end

    url = "$(provider.base_url)/$(provider.api_version)/$endpoint"
    serialized_body = isempty(body) ? UInt8[] : JSON3.write(body)
    response = HTTP.request(
        method,
        url;
        headers=_headers(provider),
        body=serialized_body,
        status_exception=false,
        http_kwargs...,
    )
    return _check_status(response)
end

function _extract_text(response::JSON3.Object)
    all_texts = String[]
    for candidate in response.candidates
        candidate_text = join([part.text for part in candidate.content.parts], "")
        push!(all_texts, candidate_text)
    end
    return all_texts
end

"""
    to_api_function_declaration(func_decl::FunctionDeclaration) -> Dict{String, Any}

Converts a FunctionDeclaration to the API-compatible dictionary format.

# Arguments
- `func_decl::FunctionDeclaration`: The function declaration to convert.

# Returns
- `Dict{String, Any}`: The API-compatible dictionary.
"""
function to_api_function_declaration(func_decl::FunctionDeclaration)
    params = Dict{String,Any}()
    params["type"] = func_decl.parameters.type

    if func_decl.parameters.description !== nothing
        params["description"] = func_decl.parameters.description
    end

    if func_decl.parameters.properties !== nothing
        params["properties"] = func_decl.parameters.properties
    end

    if func_decl.parameters.items !== nothing
        params["items"] = func_decl.parameters.items
    end

    if func_decl.parameters.required !== nothing
        params["required"] = func_decl.parameters.required
    end

    decl = Dict{String,Any}()
    decl["name"] = func_decl.name

    if func_decl.description !== nothing
        decl["description"] = func_decl.description
    end

    decl["parameters"] = params
    return decl
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
    :response_modalities => "responseModalities",
    :routing_config => "routingConfig",
    :media_resolution => "mediaResolution",
    :speech_config => "speechConfig",
    :audio_timestamp => "audioTimestamp",
    :thinking_config => "thinkingConfig",
)

"""
Extract generation config parameters from the config
"""
function _build_generation_config(config::GenerateContentConfig)
    generation_config = Dict{String,Any}()
    for (field, key) in GENERATION_CONFIG_FIELDS
        value = getfield(config, field)
        value === nothing || (generation_config[key] = value)
    end
    return generation_config
end

"""
    list_models(provider::AbstractGoogleProvider) -> Vector{Dict}
    list_models(api_key::String) -> Vector{Dict}
    list_models() -> Vector{Dict}

Retrieve a list of available models along with their details from the Google AI API.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 

# Returns
- `Vector{Dict}`: A list of dictionaries, each containing details about an available model.
"""
function list_models(provider::AbstractGoogleProvider)
    endpoint = "models"
    response = _request(provider, endpoint, :GET, Dict())
    parsed_response = JSON3.read(response.body)

    if !haskey(parsed_response, :models)
        return Vector{Dict}()
    end
    models = [
        Dict(
            :name => replace(model.name, "models/" => ""),
            :version => model.version,
            :display_name => model.displayName,
            :description => get(model, :description, nothing),
            :prompt_token_limit => model.inputTokenLimit,
            :output_token_limit => model.outputTokenLimit,
            :supported_generation_methods => model.supportedGenerationMethods,
            :temperature => get(model, :temperature, nothing),
            :topP => get(model, :topP, nothing),
            :topK => get(model, :topK, nothing),
        ) for model in parsed_response.models
    ]
    return models
end

list_models(api_key::String) = list_models(GoogleProvider(; api_key))
list_models() = list_models(GoogleProvider())
