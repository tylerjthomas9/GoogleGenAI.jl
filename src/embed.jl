function _embed_request(
    model, content; task_type=nothing, title=nothing, output_dimensionality=nothing
)
    return _compact(
        "model" => model,
        "content" => _to_content(content, nothing),
        "taskType" => task_type,
        "title" => title,
        "outputDimensionality" => output_dimensionality,
    )
end

function _embed(provider, model_name, inputs; http_kwargs=NamedTuple(), options...)
    model = _model_path(model_name)
    body = Dict("requests" => [_embed_request(model, x; options...) for x in inputs])
    response = _request(provider, "$model:batchEmbedContents", :POST, body; http_kwargs...)
    values = [Float64.(e.values) for e in JSON3.read(response.body).embeddings]
    return values, response.status
end

"""
    embed_content([provider | api_key], model_name, content; task_type=nothing, title=nothing, output_dimensionality=nothing, http_kwargs=NamedTuple()) -> NamedTuple
    embed_content([provider | api_key], model_name, contents::AbstractVector; ...) -> NamedTuple

Generate embeddings. A single `content` gives one embedding; a vector gives one embedding per
element. Each content is a string, a media NamedTuple or file (for multimodal embedding
models), or a content dict with `parts` to embed several parts together.

# Keyword Arguments
- `task_type`: What the embedding is for, e.g. `"RETRIEVAL_QUERY"`, `"RETRIEVAL_DOCUMENT"`,
  `"SEMANTIC_SIMILARITY"`, `"CLASSIFICATION"` or `"CLUSTERING"`.
- `title`: Document title, used with `task_type="RETRIEVAL_DOCUMENT"`.
- `output_dimensionality`: Truncate embeddings to this many dimensions.
- `http_kwargs`: Keyword arguments passed to `HTTP.request`.

# Returns
- `values`: The embedding as a `Vector{Float64}`, or one such vector per input.
- `response_status`: The HTTP status code.
"""
function embed_content(
    provider::AbstractGoogleProvider, model_name::String, content; kwargs...
)
    values, status = _embed(provider, model_name, (content,); kwargs...)
    return (values=only(values), response_status=status)
end

function embed_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents::AbstractVector;
    kwargs...,
)
    values, status = _embed(provider, model_name, contents; kwargs...)
    return (values=values, response_status=status)
end

function embed_content(api_key::String, model_name::String, contents; kwargs...)
    return embed_content(GoogleProvider(; api_key), model_name, contents; kwargs...)
end

function embed_content(model_name::String, contents; kwargs...)
    return embed_content(GoogleProvider(), model_name, contents; kwargs...)
end
