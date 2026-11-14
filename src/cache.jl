_timestamp(t::AbstractString) = t
_timestamp(t::DateTime) = Dates.format(t, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

function _expiration(ttl, expire_time)
    ttl === nothing ||
        expire_time === nothing ||
        throw(ArgumentError("set only one of `ttl` and `expire_time`"))
    expire_time === nothing ||
        return Dict{String,Any}("expireTime" => _timestamp(expire_time))
    return _compact("ttl" => ttl)
end

"""
    create_cached_content([provider | api_key], model_name, content; ttl=nothing, expire_time=nothing, display_name=nothing, system_instruction=nothing, tools=nothing, tool_config=nothing, http_kwargs=NamedTuple()) -> JSON3.Object

Create a cached content resource that can be reused in later requests through
`GenerateContentConfig(cached_content=...)`.

# Arguments
- `model_name::String`: The model to use (e.g. "gemini-3.5-flash-lite").
- `content`: What to cache: anything [`generate_content`](@ref) accepts as `contents`.

# Keyword Arguments
- `ttl::String`: How long to keep the cache, e.g. `"300s"`. Defaults to `"300s"` when
  `expire_time` is not set either.
- `expire_time`: When the cache expires, as a UTC `DateTime` or an RFC 3339 string.
- `display_name::String`: A human-readable name for the cache.
- `system_instruction`: A system instruction to cache, in any form `GenerateContentConfig` accepts.
- `tools`, `tool_config`: Tools to cache, in the same forms as `GenerateContentConfig`.
- `http_kwargs`: Keyword arguments passed to `HTTP.request`.

# Returns
The created cache's metadata; its `name` (e.g. `"cachedContents/abc"`) identifies it.
"""
function create_cached_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    content;
    ttl::Union{Nothing,String}=nothing,
    expire_time::Union{Nothing,DateTime,AbstractString}=nothing,
    display_name::Union{Nothing,String}=nothing,
    system_instruction=nothing,
    tools=nothing,
    tool_config::Union{Nothing,ToolConfig}=nothing,
    http_kwargs=NamedTuple(),
)
    expire_time === nothing && ttl === nothing && (ttl = "300s")
    body = merge(
        _expiration(ttl, expire_time),
        _compact(
            "model" => _model_path(model_name),
            "contents" => _to_contents(content),
            "displayName" => display_name,
            "systemInstruction" => _system_instruction(system_instruction),
            "toolConfig" => _build_tool_config(tool_config),
        ),
    )
    built_tools = _build_tools(tools)
    isempty(built_tools) || (body["tools"] = built_tools)
    return _request_json(provider, "cachedContents", :POST, body; http_kwargs...)
end

function create_cached_content(api_key::String, model_name::String, content; kwargs...)
    return create_cached_content(GoogleProvider(; api_key), model_name, content; kwargs...)
end

function create_cached_content(model_name::String, content; kwargs...)
    return create_cached_content(GoogleProvider(), model_name, content; kwargs...)
end

"""
    list_cached_content([provider | api_key]; http_kwargs=NamedTuple()) -> Vector

Lists the metadata of all your cached content, fetching all pages. The cached content itself
is not returned.
"""
function list_cached_content(provider::AbstractGoogleProvider; http_kwargs=NamedTuple())
    return _paginate(
        provider, "cachedContents", :cachedContents; page_size=1000, http_kwargs...
    )
end

function list_cached_content(api_key::String; kwargs...)
    return list_cached_content(GoogleProvider(; api_key); kwargs...)
end
list_cached_content(; kwargs...) = list_cached_content(GoogleProvider(); kwargs...)

"""
    get_cached_content([provider | api_key], cache_name; http_kwargs=NamedTuple()) -> JSON3.Object

Retrieve the metadata for one cached content resource by its name (e.g. "cachedContents/12345").
"""
function get_cached_content(
    provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()
)
    return _request_json(provider, cache_name, :GET; http_kwargs...)
end

function get_cached_content(api_key::String, cache_name::String; kwargs...)
    return get_cached_content(GoogleProvider(; api_key), cache_name; kwargs...)
end

function get_cached_content(cache_name::String; kwargs...)
    return get_cached_content(GoogleProvider(), cache_name; kwargs...)
end

"""
    update_cached_content(provider, cache_name, ttl=nothing; expire_time=nothing, http_kwargs=NamedTuple()) -> JSON3.Object
    update_cached_content(api_key, cache_name, ttl; expire_time=nothing, http_kwargs=NamedTuple()) -> JSON3.Object
    update_cached_content(cache_name, ttl=nothing; expire_time=nothing, http_kwargs=NamedTuple()) -> JSON3.Object

Change when a cache expires: pass a new `ttl` (e.g. `"600s"`) or an `expire_time` (UTC
`DateTime` or RFC 3339 string). Other fields cannot be updated. Returns the updated metadata.
"""
function update_cached_content(
    provider::AbstractGoogleProvider,
    cache_name::String,
    ttl::Union{Nothing,String}=nothing;
    expire_time::Union{Nothing,DateTime,AbstractString}=nothing,
    http_kwargs=NamedTuple(),
)
    body = _expiration(ttl, expire_time)
    isempty(body) && throw(ArgumentError("pass a `ttl` or an `expire_time`"))
    return _request_json(provider, cache_name, :PATCH, body; http_kwargs...)
end

function update_cached_content(
    api_key::String, cache_name::String, ttl::Union{Nothing,String}; kwargs...
)
    return update_cached_content(GoogleProvider(; api_key), cache_name, ttl; kwargs...)
end

function update_cached_content(
    cache_name::String, ttl::Union{Nothing,String}=nothing; kwargs...
)
    return update_cached_content(GoogleProvider(), cache_name, ttl; kwargs...)
end

"""
    delete_cached_content([provider | api_key], cache_name; http_kwargs=NamedTuple()) -> Int

Delete a cached content resource by its name and return the HTTP status code.
"""
function delete_cached_content(
    provider::AbstractGoogleProvider, cache_name::String; http_kwargs=NamedTuple()
)
    return _request(provider, cache_name, :DELETE; http_kwargs...).status
end

function delete_cached_content(api_key::String, cache_name::String; kwargs...)
    return delete_cached_content(GoogleProvider(; api_key), cache_name; kwargs...)
end

function delete_cached_content(cache_name::String; kwargs...)
    return delete_cached_content(GoogleProvider(), cache_name; kwargs...)
end
