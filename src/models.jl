function _model_info(model)
    return Dict(
        :name => replace(model.name, "models/" => ""),
        :version => get(model, :version, nothing),
        :display_name => get(model, :displayName, nothing),
        :description => get(model, :description, nothing),
        :prompt_token_limit => get(model, :inputTokenLimit, nothing),
        :output_token_limit => get(model, :outputTokenLimit, nothing),
        :supported_generation_methods => get(model, :supportedGenerationMethods, []),
        :thinking => get(model, :thinking, nothing),
        :temperature => get(model, :temperature, nothing),
        :max_temperature => get(model, :maxTemperature, nothing),
        :topP => get(model, :topP, nothing),
        :topK => get(model, :topK, nothing),
    )
end

"""
    list_models([provider | api_key]; http_kwargs=NamedTuple()) -> Vector{Dict}

Retrieve every available model (all pages) along with its details.

Each entry has `:name`, `:version`, `:display_name`, `:description`, `:prompt_token_limit`,
`:output_token_limit`, `:supported_generation_methods`, `:thinking`, `:temperature`,
`:max_temperature`, `:topP` and `:topK`.
"""
function list_models(provider::AbstractGoogleProvider; http_kwargs=NamedTuple())
    models = _paginate(provider, "models", :models; page_size=1000, http_kwargs...)
    return [_model_info(model) for model in models]
end

list_models(api_key::String; kwargs...) = list_models(GoogleProvider(; api_key); kwargs...)
list_models(; kwargs...) = list_models(GoogleProvider(); kwargs...)

"""
    get_model([provider | api_key], model_name; http_kwargs=NamedTuple()) -> Dict

Retrieve one model's details, in the same shape as [`list_models`](@ref) entries.
"""
function get_model(
    provider::AbstractGoogleProvider, model_name::String; http_kwargs=NamedTuple()
)
    return _model_info(
        _request_json(provider, _model_path(model_name), :GET; http_kwargs...)
    )
end

function get_model(api_key::String, model_name::String; kwargs...)
    return get_model(GoogleProvider(; api_key), model_name; kwargs...)
end
function get_model(model_name::String; kwargs...)
    return get_model(GoogleProvider(), model_name; kwargs...)
end
