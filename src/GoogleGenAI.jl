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
    error("Request failed with status $(resp.status) $(resp.message) $logs")
end

function _request(
    provider::AbstractGoogleProvider,
    endpoint::String,
    method::Symbol,
    body::Dict;
    http_kwargs...
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
            candidates = [],
            safety_ratings = Dict(),
            text = "",
            response_status = response.status,
            finish_reason = "UNKNOWN",
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
    generate_content(provider::AbstractGoogleProvider, model_name::String, prompt::String, image_path::String; api_kwargs=NamedTuple(), http_kwargs=NamedTuple()) -> NamedTuple
    generate_content(api_key::String, model_name::String, prompt::String, image_path::String; api_kwargs=NamedTuple(), http_kwargs=NamedTuple()) -> NamedTuple
    
    generate_content(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; api_kwargs=NamedTuple(), http_kwargs=NamedTuple()) -> NamedTuple
    generate_content(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; api_kwargs=NamedTuple(), http_kwargs=NamedTuple()) -> NamedTuple

Generate content based on a combination of text prompt and an image (optional).

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt to accompany the image.
- `image_path::String` (optional): The path to the image file to include in the request.

# API Keyword Arguments
- `temperature::Float64` (optional): Controls the randomness in the generation process. Higher values result in more random outputs. Typically ranges between 0 and 1.
- `candidate_count::Int` (optional): The number of generation candidates to consider. Currently, only one candidate can be specified.
- `max_output_tokens::Int` (optional): The maximum number of tokens that the generated content should contain.
- `stop_sequences::Vector{String}` (optional): A list of sequences where the generation should stop. Useful for defining natural endpoints in generated content.
- `safety_settings::Vector{Dict}` (optional): Settings to control the safety aspects of the generated content, such as filtering out unsafe or inappropriate content.

# HTTP Kwargs
- All keyword arguments supported by the `HTTP.request` function. Documentation can be found here: https://juliaweb.github.io/HTTP.jl/stable/reference/#HTTP.request.

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
    prompt::String;
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    endpoint = "models/$model_name:generateContent"

    safety_settings = get(api_kwargs, :safety_settings, nothing)
    generation_config = Dict{String,Any}()
    for key in keys(api_kwargs)
        if key != :safety_settings
            generation_config[string(key)] = getproperty(api_kwargs, key)
        end
    end

    body = Dict(
        "contents" => [Dict("parts" => [Dict("text" => prompt)])],
        "generationConfig" => generation_config,
        "safetySettings" => safety_settings,
    )

    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    return _parse_response(response)
end

function generate_content(
    api_key::String,
    model_name::String,
    prompt::String;
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, prompt; api_kwargs, http_kwargs
    )
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String,
    image_path::String;
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    image_data = open(base64encode, image_path)
    safety_settings = get(api_kwargs, :safety_settings, nothing)

    generation_config = Dict{String,Any}()
    for key in keys(api_kwargs)
        if key != :safety_settings
            generation_config[string(key)] = getproperty(api_kwargs, key)
        end
    end

    body = Dict(
        "contents" => [
            Dict(
                "parts" => [
                    Dict("text" => prompt),
                    Dict("inline_data" => Dict("mime_type" => "image/jpeg", "data" => image_data)),
                ],
            ),
        ],
        "generationConfig" => generation_config,
        "safetySettings" => safety_settings,
    )

    response = _request(provider, "models/$model_name:generateContent", :POST, body; http_kwargs...)
    return _parse_response(response)
end

function generate_content(
    api_key::String,
    model_name::String,
    prompt::String,
    image_path::String;
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, prompt, image_path; api_kwargs, http_kwargs
    )
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    endpoint = "models/$model_name:generateContent"

    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    safety_settings = get(api_kwargs, :safety_settings, nothing)
    generation_config = Dict{String,Any}()
    for key in keys(api_kwargs)
        if key != :safety_settings
            generation_config[string(key)] = getproperty(api_kwargs, key)
        end
    end

    body = Dict(
        "contents" => contents,
        "generationConfig" => generation_config,
        "safetySettings" => safety_settings,
    )

    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    return _parse_response(response)
end

function generate_content(
    api_key::String,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, conversation; api_kwargs, http_kwargs
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
    embed_content(provider::AbstractGoogleProvider, model_name::String, prompt::String http_kwargs=NamedTuple()) -> NamedTuple
    embed_content(api_key::String, model_name::String, prompt::String http_kwargs=NamedTuple()) -> NamedTuple
    embed_content(provider::AbstractGoogleProvider, model_name::String, prompts::Vector{String} http_kwargs=NamedTuple()) -> Vector{NamedTuple}
    embed_content(api_key::String, model_name::String, prompts::Vector{String}, http_kwargs=NamedTuple()) -> Vector{NamedTuple}

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
    - `values`: A vector of `Float64` representing the embedding values for the given prompt.
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
        get(JSON3.read(response.body), "embedding", Dict()),
        "values",
        Vector{Float64}(),
    )
    return (values=embedding_values, response_status=response.status)
end

function embed_content(
    api_key::String,
    model_name::String,
    prompt::String;
    http_kwargs=NamedTuple(),
)
    return embed_content(
        GoogleProvider(; api_key), model_name, prompt; http_kwargs...
    )
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
    api_key::String,
    model_name::String,
    prompts::Vector{String};
    http_kwargs=NamedTuple(),
)
    return embed_content(
        GoogleProvider(; api_key), model_name, prompts; http_kwargs...
    )
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
    ) -> NamedTuple

Create a cached content resource that can be reused in subsequent requests.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests
- `model_name::String`: The model to use (e.g. "gemini-1.5-flash-001")
- `content`: Content to cache (string, conversation array, or raw content dict)
- `ttl`: Time-to-live duration for the cache (default "300s")
- `system_instruction`: Optional system instruction for the model
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
        "model" => "models/$model_name",
        "contents" => contents,
        "ttl" => ttl,
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

Lists the cache metadata for all your cached content. 
(Does not return or expose the cached content itself.)
"""
function list_cached_content(
    provider::AbstractGoogleProvider;
    http_kwargs=NamedTuple(),
)
    endpoint = "cachedContents"
    response = _request(provider, endpoint, :GET, Dict(); http_kwargs...)
    parsed = JSON3.read(response.body)

    return parsed[:cachedContents]
end

function list_cached_content(
    api_key::String;
    http_kwargs=NamedTuple(),
)
    return list_cached_content(GoogleProvider(; api_key); http_kwargs...)
end

"""
    get_cached_content(provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()) -> JSON3.Object

Retrieves metadata for a single cached content by its name.
Example: 
    cache_name = "cachedContents/12345"
"""
function get_cached_content(
    provider::AbstractGoogleProvider,
    cache_name::String;
    http_kwargs=NamedTuple(),
)
    # The resource name is the entire "cachedContents/..." path
    response = _request(provider, cache_name, :GET, Dict(); http_kwargs...)
    return JSON3.read(response.body)
end

function get_cached_content(
    api_key::String,
    cache_name::String;
    http_kwargs=NamedTuple(),
)
    return get_cached_content(
        GoogleProvider(; api_key), cache_name;
        http_kwargs...
    )
end

"""
    update_cached_content(provider::AbstractGoogleProvider, cache_name::String; ttl="600s") -> Dict

Updates the TTL of an existing cache. 
Any attempt to change other fields is not supported.
Example usage:
    update_cached_content(provider, "cachedContents/xyz123"; ttl="600s")
"""
function update_cached_content(
    provider::AbstractGoogleProvider,
    cache_name::String;
    ttl::String="600s",
    http_kwargs=NamedTuple(),
)
    # This is a PATCH request to the exact resource name
    # The body can include 'ttl' (or 'expireTime')
    body = Dict("ttl" => ttl)
    response = _request(provider, cache_name, :PATCH, body; http_kwargs...)
    return JSON3.read(response.body)
end

function update_cached_content(
    api_key::String,
    cache_name::String;
    ttl::String="600s",
    http_kwargs=NamedTuple(),
)
    return update_cached_content(
        GoogleProvider(; api_key), cache_name; ttl=ttl, http_kwargs...
    )
end

"""
    delete_cached_content(provider::AbstractGoogleProvider, cache_name::String) -> Nothing

Deletes a cached content resource.
Example usage:
    delete_cached_content(provider, "cachedContents/xyz123")
"""
function delete_cached_content(
    provider::AbstractGoogleProvider,
    cache_name::String;
    http_kwargs=NamedTuple(),
)
    response = _request(provider, cache_name, :DELETE, Dict(); http_kwargs...)
    return response.status
end

function delete_cached_content(
    api_key::String,
    cache_name::String;
    http_kwargs=NamedTuple(),
)
    return delete_cached_content(GoogleProvider(; api_key), cache_name; http_kwargs...)
end


"""
    generate_content_with_cache(
        provider::AbstractGoogleProvider,
        model_name::String,
        prompt::Union{String,Vector{Dict{Symbol,Any}}};
        cached_content::String,
        http_kwargs=NamedTuple(),
        api_kwargs=NamedTuple()
    ) -> NamedTuple

Generate new content while referencing an existing cache. The `cached_content`
argument should be the full cache resource name, e.g. "cachedContents/12345".

The prompt can be either a string for single-turn generation, or a vector of 
conversation messages for multi-turn generation.
"""
function generate_content_with_cache(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::Union{String,Vector{Dict{Symbol,Any}}};
    cached_content::String,
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    endpoint = "models/$model_name:generateContent"

    generation_config = Dict{String,Any}()
    for k in keys(api_kwargs)
        generation_config[string(k)] = getproperty(api_kwargs, k)
    end

    # Handle both string prompts and conversation arrays
    contents = if prompt isa String
        [Dict("parts" => [Dict("text" => prompt)], "role" => "user")]
    else
        prompt
    end

    body = Dict(
        "contents" => contents,
        "cachedContent" => cached_content,
        "generationConfig" => generation_config,
    )

    response = _request(provider, endpoint, :POST, body; http_kwargs...)
    return _parse_response(response)
end

"""
    generate_content_with_cache(
        api_key::String,
        model_name::String,
        prompt::Union{String,Vector{Dict{Symbol,Any}}};
        cached_content::String,
        api_kwargs=NamedTuple(),
        http_kwargs=NamedTuple(),
    ) -> NamedTuple

Same as above, but accepts `api_key` instead of a provider.
"""
function generate_content_with_cache(
    api_key::String,
    model_name::String,
    prompt::Union{String,Vector{Dict{Symbol,Any}}};
    cached_content::String,
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    return generate_content_with_cache(
        GoogleProvider(; api_key),
        model_name,
        prompt;
        cached_content=cached_content,
        api_kwargs=api_kwargs,
        http_kwargs=http_kwargs,
    )
end

##
# NEW method to match calls like:
#    generate_content_with_cache(secret_key, model, cache_name, single_prompt)
# i.e. 4 positional arguments: (api_key, model_name, cache_name, prompt)
##

function generate_content_with_cache(
    api_key::String,
    model_name::String,
    cache_name::String,
    prompt::Union{String,Vector{Dict{Symbol,Any}}};
    api_kwargs=NamedTuple(),
    http_kwargs=NamedTuple(),
)
    return generate_content_with_cache(
        GoogleProvider(; api_key),
        model_name,
        prompt;
        cached_content=cache_name,
        api_kwargs=api_kwargs,
        http_kwargs=http_kwargs,
    )
end

export GoogleProvider,
       generate_content,
       generate_content_with_cache,
       count_tokens,
       embed_content,
       list_models,
       create_cached_content,
       list_cached_content,
       get_cached_content,
       update_cached_content,
       delete_cached_content

end # module GoogleGenAI
