
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
