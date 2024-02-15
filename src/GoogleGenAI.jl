module GoogleGenAI

using JSON3
using HTTP

Base.@kwdef struct GoogleProvider
    api_key::String = ""
    base_url::String = "https://generativelanguage.googleapis.com/v1beta"
end

struct GoogleTextResponse
    candidates::Vector{Dict{Symbol,Any}}
    safety_ratings::Dict{Pair{Symbol,String},Pair{Symbol,String}}
    text::String
end

struct GoogleEmbeddingResponse
    values::Vector{Float64}
end

#TODO: Add support for exception
struct BlockedPromptException <: Exception end

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
    safety_rating = Dict(parsed_response.promptFeedback.safetyRatings)
    return GoogleTextResponse(candidates, safety_rating, concatenated_texts)
end

#TODO: Add Documentation and tests (this is from the python api)
# temperature: The temperature for randomness in generation. Defaults to None.
# candidate_count: The number of candidates to consider. Defaults to None.
# max_output_tokens: The maximum number of output tokens. Defaults to None.
# top_p: The nucleus sampling probability threshold. Defaults to None.
# top_k: The top-k sampling parameter. Defaults to None.
# safety_settings: Safety settings for generated text. Defaults to None.
# stop_sequences: Stop sequences to halt text generation. Can be a string
#         or iterable of strings. Defaults to None.
function generate_content(
    provider::GoogleProvider, model_name::String, input::String; kwargs...
)
    url = "$(provider.base_url)/models/$model_name:generateContent?key=$(provider.api_key)"
    generation_config = Dict{String,Any}()
    for (key, value) in kwargs
        generation_config[string(key)] = value
    end

    body = Dict(
        "contents" => [Dict("parts" => [Dict("text" => input)])],
        "generationConfig" => generation_config,
    )
    response = HTTP.post(
        url; headers=Dict("Content-Type" => "application/json"), body=JSON3.write(body)
    )
    if response.status >= 200 && response.status < 300
        return _parse_response(response)
    else
        error("Request failed with status $(response.status): $(String(response.body))")
    end
end
function generate_content(api_key::String, model_name::String, input::String; kwargs...)
    return generate_content(GoogleProvider(; api_key), model_name, input; kwargs...)
end

function count_tokens(provider::GoogleProvider, model_name::String, input::String)
    url = "$(provider.base_url)/models/$model_name:countTokens?key=$(provider.api_key)"
    body = Dict("contents" => [Dict("parts" => [Dict("text" => input)])])
    response = HTTP.post(
        url; headers=Dict("Content-Type" => "application/json"), body=JSON3.write(body)
    )

    if response.status >= 200 && response.status < 300
        parsed_response = JSON3.read(response.body)
        total_tokens = get(parsed_response, "totalTokens")
        return total_tokens
    else
        error("Request failed with status $(response.status): $(String(response.body))")
    end
end
function count_tokens(api_key::String, model_name::String, input::String)
    return count_tokens(GoogleProvider(; api_key), model_name, input)
end

#TODO: Do we want an embeddings struct, or just the array of embeddings?
function embed_content(provider::GoogleProvider, model_name::String, input::String)
    url = "$(provider.base_url)/models/$model_name:embedContent?key=$(provider.api_key)"
    body = Dict(
        "model" => "models/$model_name",
        "content" => Dict("parts" => [Dict("text" => input)]),
    )
    response = HTTP.post(
        url; headers=Dict("Content-Type" => "application/json"), body=JSON3.write(body)
    )

    if response.status >= 200 && response.status < 300
        parsed_response = JSON3.read(response.body)
        embedding_values = get(
            get(parsed_response, "embedding", Dict()), "values", Vector{Float64}()
        )
        return GoogleEmbeddingResponse(embedding_values)
    else
        error("Request failed with status $(response.status): $(String(response.body))")
    end
end
function embed_content(api_key::String, model_name::String, input::String)
    return embed_content(GoogleProvider(; api_key), model_name, input)
end

function list_models(provider::GoogleProvider)
    url = "$(provider.base_url)/models?key=$(provider.api_key)"

    response = HTTP.get(url; headers=Dict("Content-Type" => "application/json"))

    if response.status >= 200 && response.status < 300
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
    else
        error("Request failed with status $(response.status): $(String(response.body))")
    end
end
list_models(api_key::String) = list_models(GoogleProvider(; api_key))

export GoogleProvider, generate_content, count_tokens, embed_content, list_models

end # module GoogleGenAI
