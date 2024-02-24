module GoogleGenAI

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
struct GoogleTextResponse
    candidates::Vector{Dict{Symbol,Any}}
    safety_ratings::Dict{Pair{Symbol,String},Pair{Symbol,String}}
    text::String
    response_status::Int
    finish_reason::String
end

struct GoogleEmbeddingResponse
    values::Vector{Float64}
    response_status::Int
end

#TODO: Add support for exception
struct BlockedPromptException <: Exception end

function status_error(resp, log=nothing)
    logs = !isnothing(log) ? ": $log" : ""
    return error("Request failed with status $(resp.status) $(resp.message)$logs")
end

function _request(
    provider::AbstractGoogleProvider, endpoint::String, method::Symbol, body::Dict
)
    url = "$(provider.base_url)/$(provider.api_version)/$endpoint?key=$(provider.api_key)"
    headers = Dict("Content-Type" => "application/json")
    serialized_body = isempty(body) ? UInt8[] : JSON3.write(body)
    response = HTTP.request(method, url; headers=headers, body=serialized_body)
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
    all_texts = _extract_text(parsed_response)
    concatenated_texts = join(all_texts, "")
    candidates = [Dict(i) for i in parsed_response[:candidates]]
    finish_reason = candidates[end][:finishReason]
    safety_rating = Dict(parsed_response.promptFeedback.safetyRatings)
    return GoogleTextResponse(
        candidates, safety_rating, concatenated_texts, response.status, finish_reason
    )
end

"""
    generate_content(provider::GooglePAbstractGoogleProviderrovider, model_name::String, input::String; kwargs...) -> GoogleTextResponse
    generate_content(api_key::String, model_name::String, input::String; kwargs...) -> GoogleTextResponse

Generate text using the specified model.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `input::String`: The input prompt based on which the text is generated.

# Keyword Arguments
- `temperature::Float64` (optional): Controls the randomness in the generation process. Higher values result in more random outputs. Typically ranges between 0 and 1.
- `candidate_count::Int` (optional): The number of generation candidates to consider. Currently, only one candidate can be specified.
- `max_output_tokens::Int` (optional): The maximum number of tokens that the generated content should contain.
- `stop_sequences::Vector{String}` (optional): A list of sequences where the generation should stop. Useful for defining natural endpoints in generated content.
- `safety_settings::Vector{Dict}` (optional): Settings to control the safety aspects of the generated content, such as filtering out unsafe or inappropriate content.

# Returns
- `GoogleTextResponse`
"""
function generate_content(
    provider::AbstractGoogleProvider, model_name::String, input::String; kwargs...
)
    endpoint = "models/$model_name:generateContent"

    generation_config = Dict{String,Any}()
    for (key, value) in kwargs
        if key != :safety_settings
            generation_config[string(key)] = value
        end
    end

    if haskey(kwargs, :safety_settings)
        safety_settings = kwargs[:safety_settings]
    else
        safety_settings = nothing
    end
    body = Dict(
        "contents" => [Dict("parts" => [Dict("text" => input)])],
        "generationConfig" => generation_config,
        "safetySettings" => safety_settings,
    )

    response = _request(provider, endpoint, :POST, body)
    return _parse_response(response)
end
function generate_content(api_key::String, model_name::String, input::String; kwargs...)
    return generate_content(GoogleProvider(; api_key), model_name, input; kwargs...)
end

"""
    count_tokens(provider::AbstractGoogleProvider, model_name::String, input::String) -> Int
    count_tokens(api_key::String, model_name::String, input::String) -> Int

Calculate the number of tokens generated by the specified model for a given input string.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `input::String`: The input prompt based on which the text is generated.

# Returns
- `Int`: The total number of tokens that the given input string would be broken into by the specified model's tokenizer.
"""
function count_tokens(provider::AbstractGoogleProvider, model_name::String, input::String)
    endpoint = "models/$model_name:countTokens"
    body = Dict("contents" => [Dict("parts" => [Dict("text" => input)])])
    response = _request(provider, endpoint, :POST, body)
    total_tokens = get(JSON3.read(response.body), "totalTokens", 0)
    return total_tokens
end
function count_tokens(api_key::String, model_name::String, input::String)
    return count_tokens(GoogleProvider(; api_key), model_name, input)
end

#TODO: Do we want an embeddings struct, or just the array of embeddings?
"""
    embed_content(provider::AbstractGoogleProvider, model_name::String, input::String) -> GoogleEmbeddingResponse
    embed_content(api_key::String, model_name::String, input::String) -> GoogleEmbeddingResponse

Generate an embedding for the given input text using the specified model.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `input::String`: The input prompt based on which the text is generated.

# Returns
- `GoogleEmbeddingResponse`
"""
function embed_content(provider::AbstractGoogleProvider, model_name::String, input::String)
    endpoint = "models/$model_name:embedContent"
    body = Dict(
        "model" => "models/$model_name",
        "content" => Dict("parts" => [Dict("text" => input)]),
    )
    response = _request(provider, endpoint, :POST, body)
    embedding_values = get(
        get(JSON3.read(response.body), "embedding", Dict()), "values", Vector{Float64}()
    )
    return GoogleEmbeddingResponse(embedding_values, response.status)
end
function embed_content(api_key::String, model_name::String, input::String)
    return embed_content(GoogleProvider(; api_key), model_name, input)
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
    models = [
        Dict(
            :name => replace(model.name, "models/" => ""),
            :version => model.version,
            :display_name => model.displayName,
            :description => model.description,
            :input_token_limit => model.inputTokenLimit,
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

export GoogleProvider, generate_content, count_tokens, embed_content, list_models

end # module GoogleGenAI
