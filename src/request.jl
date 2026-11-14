# The API key goes in a header, never the URL, so it stays out of logs and errors.
function _headers(provider::AbstractGoogleProvider)
    return Dict("Content-Type" => "application/json", "x-goog-api-key" => provider.api_key)
end

function _url(provider::AbstractGoogleProvider, endpoint)
    return "$(provider.base_url)/$(provider.api_version)/$endpoint"
end

_retry_options(::AbstractGoogleProvider) = nothing
_retry_options(provider::GoogleProvider) = provider.retry_options
_http_options(::AbstractGoogleProvider) = (;)
_http_options(provider::GoogleProvider) = provider.http_options

function _with_retries(f, provider::AbstractGoogleProvider)
    options = _retry_options(provider)
    attempt = 1
    while true
        try
            return f()
        catch e
            err = e isa HTTP.RequestError ? e.error : e
            retryable =
                options !== nothing &&
                err isa GeminiAPIError &&
                attempt < options.attempts &&
                err.code in options.http_status_codes
            retryable || rethrow()
            sleep(_retry_delay(options, attempt, err))
            attempt += 1
        end
    end
end

function _retry_delay(options::RetryOptions, attempt::Integer, e::GeminiAPIError)
    delay = something(
        e.retry_delay,
        options.initial_delay * options.exp_base^(attempt - 1) + rand() * options.jitter,
    )
    return min(delay, options.max_delay)
end

function _send(
    provider::AbstractGoogleProvider, method, url; headers=_headers(provider), kwargs...
)
    return _with_retries(provider) do
        response = HTTP.request(
            method,
            url;
            headers,
            _http_options(provider)...,
            kwargs...,
            status_exception=false,
        )
        response.status >= 400 && throw(GeminiAPIError(response))
        return response
    end
end

function _request(
    provider::AbstractGoogleProvider,
    endpoint::String,
    method::Symbol,
    body=nothing;
    kwargs...,
)
    payload = body === nothing || isempty(body) ? UInt8[] : JSON3.write(body)
    return _send(provider, method, _url(provider, endpoint); body=payload, kwargs...)
end

_request_json(args...; kwargs...) = JSON3.read(_request(args...; kwargs...).body)

function _paginate(
    provider::AbstractGoogleProvider,
    endpoint::String,
    key::Symbol;
    page_size=nothing,
    kwargs...,
)
    items = Any[]
    size_query =
        page_size === nothing ? Pair{String,String}[] : ["pageSize" => string(page_size)]
    query = size_query
    while true
        page = _request_json(provider, endpoint, :GET; query, kwargs...)
        append!(items, get(page, key, ()))
        token = get(page, :nextPageToken, "")
        isempty(token) && return items
        query = [size_query; "pageToken" => token]
    end
end
