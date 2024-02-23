module GoogleGenAI

using JSON3
using HTTP

abstract type AbstractGoogleProvider end

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
generate_content(provider::GoogleProvider, model_name::String, input::String; kwargs...)
generate_content(api_key::String, model_name::String, input::String; kwargs...)

Generate text using the specified model.

# Arguments
- `provider::GoogleProvider`: The provider to use for the request.
- `model_name::String`: The model to use for the request.
- `input::String`: The input prompt to use for the request.

# Keyword Arguments
- `temperature::Float64`: The temperature for randomness in generation. 
- `candidate_count::Int`: The number of candidates to consider. (Only one can be specified right now)
- `max_output_tokens::Int`: The maximum number of output tokens.
- `stop_sequences::Vector{String}`: Stop sequences to halt text generation.
- `safety_settings::Vector{Dict}`: Safety settings for generated text.
"""
function generate_content(
    provider::GoogleProvider, model_name::String, input::String; kwargs...
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
    println([Dict("parts" => [Dict("text" => input)])])
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

function count_tokens(provider::GoogleProvider, model_name::String, input::String)
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
function embed_content(provider::GoogleProvider, model_name::String, input::String)
    endpoint = "models/$model_name:embedContent"
    body = Dict(
        "model" => "models/$model_name",
        "content" => Dict("parts" => [Dict("text" => input)]),
    )
    response = _request(provider, endpoint, :POST, body)
    embedding_values = get(
        get(JSON3.read(response.body), "embedding", Dict()), "values", Vector{Float64}()
    )
    return GoogleEmbeddingResponse(embedding_values)
end
function embed_content(api_key::String, model_name::String, input::String)
    return embed_content(GoogleProvider(; api_key), model_name, input)
end

function list_models(provider::GoogleProvider)
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
