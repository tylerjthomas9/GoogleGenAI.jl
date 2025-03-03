module GoogleGenAI

using Base64
using JSON3
using HTTP

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
- `safety_settings::Union{Nothing,Vector{Dict{Symbol,Any}}}`: Safety settings to block unsafe content.
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
    safety_settings::Union{Nothing,Vector{Dict{Symbol,Any}}} = nothing
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

function _parse_response(response::HTTP.Messages.Response)
    parsed_response = JSON3.read(response.body)

    # If there's no "candidates" key, just return a fallback
    if !haskey(parsed_response, :candidates)
        return (
            candidates=[],
            safety_ratings=Dict(),
            text="",
            response_status=response.status,
            finish_reason="UNKNOWN",
        )
    end

    all_texts = _extract_text(parsed_response)
    concatenated_texts = join(all_texts, "")
    candidates = [Dict(i) for i in parsed_response[:candidates]]
    finish_reason = candidates[end][:finishReason]

    safety_rating = if haskey(parsed_response.candidates[end], :safetyRatings)
        Dict(parsed_response.candidates[end].safetyRatings)
    else
        Dict()
    end

    return (
        candidates=candidates,
        safety_ratings=safety_rating,
        text=concatenated_texts,
        response_status=response.status,
        finish_reason=finish_reason,
    )
end

"""
Extract generation config parameters from the config
"""
function _build_generation_config(config::GenerateContentConfig)
    generation_config = Dict{String,Any}()
    for (field, key) in [
        (:temperature, "temperature"),
        (:candidate_count, "candidateCount"),
        (:max_output_tokens, "maxOutputTokens"),
        (:stop_sequences, "stopSequences"),
        (:response_mime_type, "responseMimeType"),
        (:response_schema, "responseSchema"),
    ]
        value = getfield(config, field)
        if value !== nothing
            generation_config[key] = value
        end
    end
    return generation_config
end

"""
    generate_content(provider::AbstractGoogleProvider, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    
    generate_content(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String, config=GenerateContentConfig()) -> NamedTuple

Generate content based on a combination of text prompt and an image (optional).

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt to accompany the image.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `NamedTuple`: A named tuple containing the following keys:
    - `candidates`: A vector of dictionaries, each representing a generation candidate.
    - `safety_ratings`: A dictionary containing safety ratings for the prompt feedback.
    - `text`: A string representing the concatenated text from all candidates.
    - `response_status`: An integer representing the HTTP response status code.
    - `finish_reason`: A string indicating the reason why the generation process was finished.
"""
function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config::GenerateContentConfig=GenerateContentConfig(),
)
    endpoint = "models/$model_name:generateContent"

    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    generation_config = _build_generation_config(config)

    body = Dict("contents" => contents, "generationConfig" => generation_config)

    if config.safety_settings !== nothing
        body["safetySettings"] = config.safety_settings
    end

    if config.cached_content !== nothing
        body["cachedContent"] = config.cached_content
    end

    response = _request(provider, endpoint, :POST, body; config.http_options...)
    return _parse_response(response)
end

function generate_content(
    api_key::String,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, conversation; image_path, config
    )
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        provider,
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

function generate_content(
    api_key::String,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        GoogleProvider(; api_key),
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

"""
    generate_content_stream(provider::AbstractGoogleProvider, model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> Channel
    generate_content_stream(api_key::String, model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> Channel
    
    generate_content_stream(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String="", config=GenerateContentConfig()) -> Channel
    generate_content_stream(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String="", config=GenerateContentConfig()) -> Channel

Generate content in a streaming fashion, returning partial results as they become available.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt to accompany the image.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `Channel`: A channel that yields partial text responses as they become available.
  Each item in the channel is a named tuple with the following fields:
  - `text::String`: The partial text response.
  - `finish_reason::Union{String,Nothing}`: The reason why generation stopped, if applicable.
  - `is_final::Bool`: Whether this is the final chunk of the response.
"""
function generate_content_stream(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config::GenerateContentConfig=GenerateContentConfig(),
)
    endpoint = "models/$model_name:streamGenerateContent"

    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    generation_config = _build_generation_config(config)

    body = Dict("contents" => contents, "generationConfig" => generation_config)

    if config.safety_settings !== nothing
        body["safetySettings"] = config.safety_settings
    end

    if config.cached_content !== nothing
        body["cachedContent"] = config.cached_content
    end

    # Create a channel to stream the results
    result_channel = Channel{NamedTuple}(32)

    # Start a task to handle the streaming
    @async begin
        try
            if isempty(provider.api_key)
                throw(ArgumentError("api_key cannot be empty"))
            end

            url = "$(provider.base_url)/$(provider.api_version)/$endpoint?alt=sse&key=$(provider.api_key)"
            headers = Dict("Content-Type" => "application/json")

            serialized_body = isempty(body) ? UInt8[] : JSON3.write(body)

            # Use a simpler approach with HTTP.get
            response = HTTP.request(
                "POST", url, headers, serialized_body; status_exception=false
            )

            if response.status >= 400
                error_msg = String(response.body)
                put!(
                    result_channel,
                    (
                        error=ErrorException(
                            "Request failed with status $(response.status): $error_msg"
                        ),
                        text="",
                        finish_reason=nothing,
                        is_final=true,
                    ),
                )
                return nothing
            end

            # Process the streaming response
            buffer = IOBuffer()
            current_text = ""
            finish_reason = nothing

            # Split the response by data: lines
            response_text = String(response.body)
            lines = split(response_text, "\n")

            for line in lines
                # Skip empty lines and SSE prefixes
                if isempty(line) || startswith(line, ":")
                    continue
                end

                # Extract the data part (SSE format: "data: {json}")
                if !startswith(line, "data:")
                    continue
                end

                data_str = strip(replace(line, r"^data:" => ""))

                # Skip empty data or end marker
                if isempty(data_str) || data_str == "[DONE]"
                    continue
                end

                # Parse the JSON data
                try
                    parsed_data = JSON3.read(data_str)

                    # Check if there are candidates
                    if haskey(parsed_data, :candidates) && !isempty(parsed_data.candidates)
                        candidate = parsed_data.candidates[1]

                        # Extract text from the candidate
                        if haskey(candidate, :content) && haskey(candidate.content, :parts)
                            chunk_text = ""
                            for part in candidate.content.parts
                                if haskey(part, :text)
                                    chunk_text *= part.text
                                end
                            end

                            # Update the current text with the new chunk
                            current_text = chunk_text

                            # Check for finish reason
                            is_final = false
                            if haskey(candidate, :finishReason) &&
                                candidate.finishReason != "FINISH_REASON_UNSPECIFIED" &&
                                candidate.finishReason != ""
                                finish_reason = candidate.finishReason
                                is_final = true
                            end

                            # Put the current chunk into the channel
                            put!(
                                result_channel,
                                (
                                    text=current_text,
                                    finish_reason=finish_reason,
                                    is_final=is_final,
                                ),
                            )

                            # If this is the final chunk, we're done
                            if is_final
                                break
                            end
                        end
                    end
                catch e
                    # Skip malformed JSON
                    @warn "Error parsing SSE data: $e"
                    continue
                end
            end

            # Ensure we always send a final chunk if we haven't already
            if !isempty(current_text) && finish_reason === nothing
                put!(
                    result_channel, (text=current_text, finish_reason="STOP", is_final=true)
                )
            end
        catch e
            # Put the error in the channel
            put!(result_channel, (error=e, text="", finish_reason=nothing, is_final=true))
        finally
            close(result_channel)
        end
    end

    return result_channel
end

function generate_content_stream(
    api_key::String,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        GoogleProvider(; api_key), model_name, conversation; image_path, config
    )
end

function generate_content_stream(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        provider,
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

function generate_content_stream(
    api_key::String,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        GoogleProvider(; api_key),
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

"""
    count_tokens(provider::AbstractGoogleProvider, model_name::String, prompt::String) -> Int
    count_tokens(api_key::String, model_name::String, prompt::String) -> Int

Calculate the number of tokens generated by the specified model for a given prompt string.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `prompt::String`: The prompt prompt based on which the text is generated.

# Returns
- `Int`: The total number of tokens that the given prompt string would be broken into by the specified model's tokenizer.
"""
function count_tokens(provider::AbstractGoogleProvider, model_name::String, prompt::String)
    endpoint = "models/$model_name:countTokens"
    body = Dict("contents" => [Dict("parts" => [Dict("text" => prompt)])])
    response = _request(provider, endpoint, :POST, body)
    total_tokens = get(JSON3.read(response.body), "totalTokens", 0)
    return total_tokens
end

function count_tokens(api_key::String, model_name::String, prompt::String)
    return count_tokens(GoogleProvider(; api_key), model_name, prompt)
end

"""
    embed_content(provider::AbstractGoogleProvider, model_name::String, prompt::String; http_kwargs=NamedTuple()) -> NamedTuple
    embed_content(api_key::String, model_name::String, prompt::String; http_kwargs=NamedTuple()) -> NamedTuple
    embed_content(provider::AbstractGoogleProvider, model_name::String, prompts::Vector{String}; http_kwargs=NamedTuple()) -> NamedTuple
    embed_content(api_key::String, model_name::String, prompts::Vector{String}; http_kwargs=NamedTuple()) -> NamedTuple

Generate an embedding for the given prompt text using the specified model.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `prompt::String`: The prompt prompt based on which the text is generated.

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `NamedTuple`: A named tuple containing the following keys:
    - `values`: A vector of `Float64` representing the embedding values for the given prompt (or prompts).
    - `response_status`: An integer representing the HTTP response status code.
"""
function embed_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    http_kwargs=NamedTuple(),
)
    endpoint = "models/$model_name:embedContent"
    body = Dict(
        "model" => "models/$model_name",
        "content" => Dict("parts" => [Dict("text" => prompt)]),
    )
    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    embedding_values = get(
        get(JSON3.read(response.body), "embedding", Dict()), "values", Vector{Float64}()
    )
    return (values=embedding_values, response_status=response.status)
end

function embed_content(
    api_key::String, model_name::String, prompt::String; http_kwargs=NamedTuple()
)
    return embed_content(GoogleProvider(; api_key), model_name, prompt; http_kwargs...)
end

"""
    embed_content(provider::AbstractGoogleProvider, model_name::String, prompts::Vector{String}; ...) -> NamedTuple
Batch embedding for multiple prompts.
"""
function embed_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompts::Vector{String};
    http_kwargs=NamedTuple(),
)
    endpoint = "models/$model_name:batchEmbedContents"
    body = Dict(
        "requests" => [
            Dict(
                "model" => "models/$model_name",
                "content" => Dict("parts" => [Dict("text" => prompt)]),
            ) for prompt in prompts
        ],
    )
    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    embedding_values = [
        get(embedding, "values", Vector{Float64}()) for
        embedding in JSON3.read(response.body)["embeddings"]
    ]
    return (values=embedding_values, response_status=response.status)
end

function embed_content(
    api_key::String, model_name::String, prompts::Vector{String}; http_kwargs=NamedTuple()
)
    return embed_content(GoogleProvider(; api_key), model_name, prompts; http_kwargs...)
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
            :description => model.description,
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

"""
    create_cached_content(
        provider::AbstractGoogleProvider,
        model_name::String,
        content::Union{String,Vector{Dict{Symbol,Any}},Dict{String,Any}};
        ttl::String="300s",
        system_instruction::String="",
        http_kwargs=NamedTuple()
    ) -> JSON3.Object
    create_cached_content(
        api_key::String,
        model_name::String,
        content::Union{String,Vector{Dict{Symbol,Any}},Dict{String,Any}};
        ttl::String="300s",
        system_instruction::String="",
        http_kwargs=NamedTuple()
    ) -> JSON3.Object

Create a cached content resource that can be reused in subsequent requests.

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: The provider instance for API requests or your Google API key as a string.
- `model_name::String`: The model to use (e.g. "gemini-1.5-flash-001").
- `content::Union{String,Vector{Dict{Symbol,Any}},Dict{String,Any}}`: The content to cache, which can be a single string, an array of conversation messages, or a raw content dictionary.
- `ttl::String`: Time-to-live duration for the cache. Defaults to `"300s"`.
- `system_instruction::String`: An optional system instruction for the model.

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `JSON3.Object`: A JSON object containing the metadata of the created cached content resource, including its cache name.
"""
function create_cached_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    content::Union{String,Vector{Dict{Symbol,Any}},Dict{String,Any}};
    ttl::String="300s",
    system_instruction::String="",
    http_kwargs=NamedTuple(),
)
    endpoint = "cachedContents"

    # Prepare the content structure
    contents = if content isa String
        [Dict("parts" => [Dict("text" => content)], "role" => "user")]
    elseif content isa Vector
        content
    else
        [content]  # Assume it's already formatted properly
    end

    body = Dict{String,Any}(
        "model" => "models/$model_name", "contents" => contents, "ttl" => ttl
    )

    if !isempty(system_instruction)
        body["systemInstruction"] = Dict("parts" => [Dict("text" => system_instruction)])
    end

    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    return JSON3.read(response.body)
end

function create_cached_content(
    api_key::String,
    model_name::String,
    content::Union{String,Vector{Dict{Symbol,Any}},Dict{String,Any}};
    ttl::String="300s",
    system_instruction::String="",
    http_kwargs=NamedTuple(),
)
    return create_cached_content(
        GoogleProvider(; api_key), model_name, content; ttl, system_instruction, http_kwargs
    )
end

"""
    list_cached_content(provider::AbstractGoogleProvider; http_kwargs=NamedTuple()) -> JSON3.Array
    list_cached_content(api_key::String; http_kwargs=NamedTuple()) -> JSON3.Array

Lists the cache metadata for all your cached content. (Does not return the cached content itself.)

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: The provider instance for API requests or your Google API key as a string.

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `JSON3.Array`: A JSON array of objects, where each object represents a cached content resource's metadata.
"""
function list_cached_content(provider::AbstractGoogleProvider; http_kwargs=NamedTuple())
    endpoint = "cachedContents"
    response = _request(provider, endpoint, :GET, Dict(); http_kwargs...)
    parsed = JSON3.read(response.body)

    return parsed[:cachedContents]
end

function list_cached_content(api_key::String; http_kwargs=NamedTuple())
    return list_cached_content(GoogleProvider(; api_key); http_kwargs...)
end

"""
    get_cached_content(provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()) -> JSON3.Object
    get_cached_content(api_key::String, cache_name::String; http_kwargs=NamedTuple()) -> JSON3.Object

Retrieve the metadata for a single cached content resource by its resource name.

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: The provider instance for API requests or your Google API key as a string.
- `cache_name::String`: The full resource name of the cached content (e.g. "cachedContents/12345").

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `JSON3.Object`: A JSON object containing the metadata for the specified cached content.
"""
function get_cached_content(
    provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()
)
    # The resource name is the entire "cachedContents/..." path
    response = _request(provider, cache_name, :GET, Dict(); http_kwargs...)
    return JSON3.read(response.body)
end

function get_cached_content(api_key::String, cache_name::String; http_kwargs=NamedTuple())
    return get_cached_content(GoogleProvider(; api_key), cache_name; http_kwargs...)
end

"""
    update_cached_content(provider::AbstractGoogleProvider, cache_name::String, ttl::String; http_kwargs=NamedTuple()) -> JSON3.Object
    update_cached_content(api_key::String, cache_name::String, ttl::String, http_kwargs=NamedTuple()) -> JSON3.Object

Update the TTL of an existing cached content resource. Attempts to change other fields are not supported.

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: The provider instance for API requests or your Google API key as a string.
- `cache_name::String`: The full resource name of the cached content (e.g. "cachedContents/xyz123").
- `ttl::String`: The new time-to-live value. Defaults to "600s".

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `JSON3.Object`: A JSON object containing the updated metadata for the cached content.
"""
function update_cached_content(
    provider::AbstractGoogleProvider,
    cache_name::String,
    ttl::String;
    http_kwargs=NamedTuple(),
)
    body = Dict("ttl" => ttl)
    response = _request(provider, cache_name, :PATCH, body; http_kwargs...)
    return JSON3.read(response.body)
end

function update_cached_content(
    api_key::String, cache_name::String, ttl::String; http_kwargs=NamedTuple()
)
    return update_cached_content(GoogleProvider(; api_key), cache_name, ttl; http_kwargs...)
end

"""
    delete_cached_content(provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()) -> Int
    delete_cached_content(api_key::String, cache_name::String; http_kwargs=NamedTuple()) -> Int

Delete a cached content resource by its resource name.

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: The provider instance for API requests or your Google API key as a string.
- `cache_name::String`: The full resource name of the cached content (e.g. "cachedContents/xyz123").

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

# Returns
- `Int`: The HTTP status code of the deletion request.
"""
function delete_cached_content(
    provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()
)
    response = _request(provider, cache_name, :DELETE, Dict(); http_kwargs...)
    return response.status
end

function delete_cached_content(
    api_key::String, cache_name::String; http_kwargs=NamedTuple()
)
    return delete_cached_content(GoogleProvider(; api_key), cache_name; http_kwargs...)
end

"""
    upload_file(provider::AbstractGoogleProvider, file_path::String; display_name::String="", mime_type::String="application/octet-stream", http_kwargs=NamedTuple()) -> JSON3.Object

Uploads a file using the media.upload endpoint. The file at `file_path` is read, base64-encoded, and sent along with optional metadata.
"""
function upload_file(
    provider::AbstractGoogleProvider,
    file_path::String;
    display_name::String="",
    mime_type::String="application/octet-stream",
    http_kwargs=NamedTuple(),
)
    # Read the file as bytes and base64 encode them
    file_bytes = read(file_path)
    file_data = base64encode(file_bytes)

    # For media uploads, use the upload endpoint (note the extra "upload/" segment)
    url = "$(provider.base_url)/upload/$(provider.api_version)/files?key=$(provider.api_key)"
    headers = Dict("Content-Type" => "application/json")

    # Build the request body with file metadata and inline data (updated key "mimeType")
    body = Dict(
        "file" => Dict(
            "displayName" => display_name,
            "mimeType" => mime_type,
            "inline_data" => Dict("data" => file_data, "mimeType" => mime_type),
        ),
    )
    serialized_body = JSON3.write(body)

    # Send the POST request
    response = HTTP.request(:POST, url, headers, serialized_body; http_kwargs...)
    if response.status >= 400
        status_error(response, String(response.body))
    end
    return JSON3.read(String(response.body))[:file]
end

# Overload for direct API key usage.
function upload_file(
    api_key::String,
    file_path::String;
    display_name::String="",
    mime_type::String="application/octet-stream",
    http_kwargs=NamedTuple(),
)
    return upload_file(
        GoogleProvider(; api_key),
        file_path;
        display_name=display_name,
        mime_type=mime_type,
        http_kwargs=http_kwargs,
    )
end

"""
    get_file(provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()) -> Any

Retrieves metadata for the file specified by its resource name (e.g. "files/abc-123").
"""
function get_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    response = _request(provider, file_name, :GET, Dict(); http_kwargs...)
    return JSON3.read(String(response.body))
end

function get_file(api_key::String, file_name::String; http_kwargs=NamedTuple())
    return get_file(GoogleProvider(; api_key), file_name; http_kwargs=http_kwargs)
end

"""
    list_files(provider::AbstractGoogleProvider; page_size::Int=10, page_token::String="", http_kwargs=NamedTuple()) -> JSON3.Array

Lists file metadata for files owned by your project. Use `page_size` and `page_token` for pagination.
"""
function list_files(
    provider::AbstractGoogleProvider;
    page_size::Int=10,
    page_token::String="",
    http_kwargs=NamedTuple(),
)
    # Build the URL with query parameters.
    url = "$(provider.base_url)/$(provider.api_version)/files?key=$(provider.api_key)&pageSize=$(page_size)"
    if page_token != ""
        url *= "&pageToken=$(page_token)"
    end
    headers = Dict("Content-Type" => "application/json")

    response = HTTP.request(:GET, url, headers, ""; http_kwargs...)
    if response.status >= 400
        status_error(response, String(response.body))
    end
    return JSON3.read(String(response.body))[:files]
end

function list_files(
    api_key::String; page_size::Int=10, page_token::String="", http_kwargs=NamedTuple()
)
    return list_files(
        GoogleProvider(; api_key);
        page_size=page_size,
        page_token=page_token,
        http_kwargs=http_kwargs,
    )
end

"""
    delete_file(provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()) -> Int

Deletes the file specified by its resource name (e.g. "files/abc-123") and returns the HTTP status code.
"""
function delete_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    response = _request(provider, file_name, :DELETE, Dict(); http_kwargs...)
    return response.status
end

function delete_file(api_key::String, file_name::String; http_kwargs=NamedTuple())
    return delete_file(GoogleProvider(; api_key), file_name; http_kwargs=http_kwargs)
end

export GoogleProvider,
    GenerateContentConfig,
    generate_content,
    generate_content_stream,
    count_tokens,
    embed_content,
    list_models,
    create_cached_content,
    list_cached_content,
    get_cached_content,
    update_cached_content,
    delete_cached_content,
    upload_file,
    get_file,
    list_files,
    delete_file

end # module GoogleGenAI
