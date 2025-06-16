
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
