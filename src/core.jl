
abstract type AbstractGoogleProvider end

"""
    Base.@kwdef struct GoogleProvider <: AbstractGoogleProvider
        api_key::String = ""
        base_url::String = "https://generativelanguage.googleapis.com"
        api_version::String = "v1beta"
    end

    A configuration object used to set up and authenticate requests to the Google Generative Language API.

    # Fields
    - `api_key::String`: Your Google API key. 
    - `base_url::String`: The base URL for the Google Generative Language API. The default is set to `"https://generativelanguage.googleapis.com"`.
    - `api_version::String`: The version of the API you wish to access. The default is set to `"v1beta"`.
"""
Base.@kwdef struct GoogleProvider <: AbstractGoogleProvider
    api_key::String = ""
    base_url::String = "https://generativelanguage.googleapis.com"
    api_version::String = "v1beta"
end

# Custom exception for blocked prompts
struct BlockedPromptException <: Exception end

function status_error(resp, log=nothing)
    logs = !isnothing(log) ? ": $log" : ""
    return error("Request failed with status $(resp.status) $(resp.message) $logs")
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
- `tools::Union{Nothing,Vector{Dict{Symbol,Any}}}`: Enables interaction with external systems.
- `tool_config::Union{Nothing,Dict{Symbol,Any}}`: Associates model output to a specific function call.
- `labels::Union{Nothing,Dict{String,String}}`: User-defined metadata labels.
- `cached_content::Union{Nothing,String}`: Resource name of a context cache.
- `response_modalities::Union{Nothing,Vector{String}}`: Requested modalities of the response.
- `media_resolution::Union{Nothing,String}`: Specified media resolution.
- `speech_config::Union{Nothing,Dict{Symbol,Any}}`: Speech generation configuration.
- `audio_timestamp::Union{Nothing,Bool}`: Whether to include audio timestamp in the request.
- `automatic_function_calling::Union{Nothing,Dict{Symbol,Any}}`: Configuration for automatic function calling.
- `thinking_config::Union{Nothing,Dict{Symbol,Any}}`: Thinking features configuration.
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
    tools::Union{Nothing,Vector{Dict{Symbol,Any}}} = nothing
    tool_config::Union{Nothing,Dict{Symbol,Any}} = nothing
    labels::Union{Nothing,Dict{String,String}} = nothing
    cached_content::Union{Nothing,String} = nothing
    response_modalities::Union{Nothing,Vector{String}} = nothing
    media_resolution::Union{Nothing,String} = nothing
    speech_config::Union{Nothing,Dict{Symbol,Any}} = nothing
    audio_timestamp::Union{Nothing,Bool} = nothing
    automatic_function_calling::Union{Nothing,Dict{Symbol,Any}} = nothing
    thinking_config::Union{Nothing,Dict{Symbol,Any}} = nothing
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

    url = "$(provider.base_url)/$(provider.api_version)/$endpoint?key=$(provider.api_key)"
    headers = Dict("Content-Type" => "application/json")

    serialized_body = isempty(body) ? UInt8[] : JSON3.write(body)
    response = HTTP.request(
        method, url; headers=headers, body=serialized_body, http_kwargs...
    )

    if response.status >= 400
        status_error(response, String(response.body))
    end
    return response
end

function _extract_text(response::JSON3.Object)
    all_texts = String[]
    for candidate in response.candidates
        candidate_text = join([part.text for part in candidate.content.parts], "")
        push!(all_texts, candidate_text)
    end
    return all_texts
end

function _parse_response(response)
    body = JSON3.read(response.body)

    text_parts = String[]
    image_parts = []
    candidates = get(body, :candidates, [])

    finish_reason = nothing
    if !isempty(candidates)
        finish_reason = get(candidates[1], :finishReason, nothing)

        content = get(candidates[1], :content, nothing)
        if content !== nothing && haskey(content, :parts)
            for part in content.parts
                if haskey(part, :text) && part.text !== nothing
                    if !isempty(strip(part.text))
                        push!(text_parts, part.text)
                    end
                elseif haskey(part, :inlineData) && part.inlineData !== nothing
                    inline_data = part.inlineData
                    mime_type = get(inline_data, :mimeType, "image/png")
                    data = get(inline_data, :data, "")

                    data = strip(data)
                    if isempty(data)
                        continue
                    end
                    image_data = Base64.base64decode(data)
                    push!(image_parts, (data=image_data, mime_type=mime_type))
                end
            end
        end
    else
        text = get(body, :text, "")
        !isempty(text) && push!(text_parts, text)
    end

    full_text = join(text_parts, "")

    return (
        candidates=candidates,
        safety_ratings=get(body, :safetyRatings, Dict{Symbol,Any}()),
        text=full_text,
        images=image_parts,
        response_status=response.status,
        finish_reason=finish_reason,
        usage_metadata=get(body, :usageMetadata, Dict{Symbol,Any}()),
    )
end

"""
Extract generation config parameters from the config
"""
function _build_generation_config(config::GenerateContentConfig)
    generation_config = Dict{String,Any}()
    config.temperature !== nothing &&
        (generation_config["temperature"] = config.temperature)
    config.top_p !== nothing && (generation_config["topP"] = config.top_p)
    config.top_k !== nothing && (generation_config["topK"] = config.top_k)
    config.candidate_count !== nothing &&
        (generation_config["candidateCount"] = config.candidate_count)
    config.max_output_tokens !== nothing &&
        (generation_config["maxOutputTokens"] = config.max_output_tokens)
    config.stop_sequences !== nothing &&
        (generation_config["stopSequences"] = config.stop_sequences)
    config.response_mime_type !== nothing &&
        (generation_config["responseMimeType"] = config.response_mime_type)
    config.response_schema !== nothing &&
        (generation_config["responseSchema"] = config.response_schema)
    config.response_modalities !== nothing &&
        (generation_config["responseModalities"] = config.response_modalities)
    config.routing_config !== nothing &&
        (generation_config["routingConfig"] = config.routing_config)
    config.media_resolution !== nothing &&
        (generation_config["mediaResolution"] = config.media_resolution)
    config.speech_config !== nothing &&
        (generation_config["speechConfig"] = config.speech_config)
    config.audio_timestamp !== nothing &&
        (generation_config["audioTimestamp"] = config.audio_timestamp)
    config.automatic_function_calling !== nothing &&
        (generation_config["automaticFunctionCalling"] = config.automatic_function_calling)
    config.thinking_config !== nothing &&
        (generation_config["thinkingConfig"] = config.thinking_config)
    return generation_config
end

"""
    list_models(provider::AbstractGoogleProvider) -> Vector{Dict}
    list_models(api_key::String) -> Vector{Dict}

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
