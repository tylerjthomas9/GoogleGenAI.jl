function _build_request_body(contents, config::GenerateContentConfig)
    body = _compact(
        "contents" => contents,
        "systemInstruction" => _system_instruction(config.system_instruction),
        "safetySettings" => config.safety_settings,
        "toolConfig" => _build_tool_config(config.tool_config),
        "cachedContent" => config.cached_content,
        "labels" => config.labels,
        "serviceTier" => config.service_tier,
    )
    generation_config = _build_generation_config(config)
    isempty(generation_config) || (body["generationConfig"] = generation_config)
    tools = _build_tools(config)
    isempty(tools) || (body["tools"] = tools)
    return body
end

function _inline_images(parts)
    images = NamedTuple{(:data, :mime_type),Tuple{Vector{UInt8},String}}[]
    for part in parts
        inline = get(part, :inlineData, nothing)
        inline === nothing && continue
        data = strip(get(inline, :data, ""))
        isempty(data) && continue
        push!(
            images,
            (
                data=base64decode(data),
                mime_type=String(get(inline, :mimeType, "image/png")),
            ),
        )
    end
    return images
end

function _parse_body(body)
    candidates = get(body, :candidates, [])
    candidate = isempty(candidates) ? nothing : first(candidates)
    content = candidate === nothing ? nothing : get(candidate, :content, nothing)
    parts = content === nothing ? () : get(content, :parts, ())
    calls = _function_calls(parts)
    # Safety ratings are per candidate; fall back to the top-level field.
    safety_ratings = if candidate !== nothing && haskey(candidate, :safetyRatings)
        candidate.safetyRatings
    else
        get(body, :safetyRatings, Dict{Symbol,Any}())
    end
    return (
        candidates=candidates,
        safety_ratings=safety_ratings,
        text=join(String(p.text) for p in parts if _is_answer_text(p)),
        images=_inline_images(parts),
        function_calls=isempty(calls) ? nothing : calls,
        finish_reason=if candidate === nothing
            nothing
        else
            get(candidate, :finishReason, nothing)
        end,
        usage_metadata=get(body, :usageMetadata, Dict{Symbol,Any}()),
    )
end

"""
    _parse_response(response) -> NamedTuple

Parse a `generateContent` HTTP response into the named tuple `generate_content` returns.
"""
function _parse_response(response)
    return (; _parse_body(JSON3.read(response.body))..., response_status=response.status)
end

"""
    generate_content([provider | api_key], model_name, contents; image_path="", images=NamedTuple[], config=GenerateContentConfig()) -> NamedTuple

Generate a response from the model.

# Arguments
- `provider::AbstractGoogleProvider` or `api_key::String`: Omit both to read the key from
  `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
- `model_name::String`: The model to use, e.g. `"gemini-3.5-flash"`.
- `contents`: A prompt string; a vector of parts for one user turn (strings, files returned by
  [`upload_file`](@ref), media NamedTuples with `path`, `data` or `uri` plus `mime_type`, or
  raw part dicts); or a conversation, i.e. a vector of content dicts with `role` and `parts`.

# Keyword Arguments
- `image_path::String`: Path of an image to attach to the last user turn.
- `images::AbstractVector`: Media NamedTuples (`path`, or `data` with `mime_type`) to attach to the last user turn.
- `config::GenerateContentConfig`: Configuration for the request.

# Returns
A named tuple with:
- `text`: The answer text (thought summaries excluded).
- `images`: Generated images as `(data, mime_type)` tuples.
- `function_calls`: `FunctionCall`s requested by the model, or `nothing`.
- `finish_reason`, `usage_metadata`, `safety_ratings`, `candidates`: As reported by the API.
- `response_status`: The HTTP status code.
"""
function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::GenerateContentConfig=GenerateContentConfig(),
)
    body = _build_request_body(_with_media(contents, image_path, images), config)
    endpoint = "$(_model_path(model_name)):generateContent"
    return _parse_response(
        _request(provider, endpoint, :POST, body; config.http_options...)
    )
end

function generate_content(api_key::String, model_name::String, contents; kwargs...)
    return generate_content(GoogleProvider(; api_key), model_name, contents; kwargs...)
end

function generate_content(model_name::String, contents; kwargs...)
    return generate_content(GoogleProvider(), model_name, contents; kwargs...)
end

function _take_lines!(buf::Vector{UInt8})
    last_newline = findlast(==(UInt8('\n')), buf)
    last_newline === nothing && return SubString{String}[]
    lines = split(String(buf[1:(last_newline - 1)]), '\n')
    deleteat!(buf, 1:last_newline)
    return lines
end

function _parse_stream_chunk(chunk_data, status)
    isempty(get(chunk_data, :candidates, ())) && return nothing
    return (; _parse_body(chunk_data)..., status=status)
end

function _error_chunk(e)
    return (
        error=e isa HTTP.RequestError ? e.error : e,
        text="",
        finish_reason="ERROR",
        is_final=true,
    )
end

"""
    generate_content_stream([provider | api_key], model_name, contents; image_path="", images=NamedTuple[], config=GenerateContentConfig()) -> Channel

Generate a response as a stream, yielding partial results as they arrive. Arguments are the
same as for [`generate_content`](@ref).

# Returns
- `Channel`: A channel that yields one named tuple per streamed chunk, with fields:
  - `text::String`: The text in this chunk.
  - `full_text::String`: All text received so far.
  - `function_calls`: The function calls received so far, or `nothing`.
  - `images`: Images in this chunk.
  - `finish_reason`: Why generation stopped; `nothing` until the last chunk.
  - `usage_metadata`: Token counts reported with this chunk.
  - `candidates`, `safety_ratings`, `status`: The raw candidates, safety ratings and HTTP status.

  If the request fails, including an error sent partway through the stream, the last
  item instead has an `error` field holding the exception (usually a
  [`GeminiAPIError`](@ref)), with `text=""` and `finish_reason="ERROR"`.
"""
function generate_content_stream(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::GenerateContentConfig=GenerateContentConfig(),
)
    body = JSON3.write(
        _build_request_body(_with_media(contents, image_path, images), config)
    )
    url = _url(provider, "$(_model_path(model_name)):streamGenerateContent")
    channel = Channel{NamedTuple}(32)

    @async try
        full_text = ""
        function_calls = FunctionCall[]
        _with_retries(provider) do
            HTTP.open(
                "POST",
                url;
                headers=_headers(provider),
                query=Dict("alt" => "sse"),
                _http_options(provider)...,
                config.http_options...,
                status_exception=false,
            ) do stream
                write(stream, body)
                HTTP.closewrite(stream)
                response = HTTP.startread(stream)
                if response.status >= 400
                    response.body = read(stream)
                    throw(GeminiAPIError(response))
                end

                pending = UInt8[]
                while !eof(stream)
                    append!(pending, readavailable(stream))
                    for line in _take_lines!(pending)
                        line = strip(line)
                        startswith(line, "data: ") || continue
                        data = SubString(line, 7)
                        data == "[DONE]" && continue
                        event = try
                            JSON3.read(data)
                        catch e
                            @warn "Failed to parse chunk: $e"
                            continue
                        end
                        if haskey(event, :error)
                            put!(
                                channel, _error_chunk(GeminiAPIError(response.status, data))
                            )
                            return nothing
                        end
                        chunk = _parse_stream_chunk(event, response.status)
                        chunk === nothing && continue
                        full_text *= chunk.text
                        chunk.function_calls === nothing ||
                            append!(function_calls, chunk.function_calls)
                        calls = isempty(function_calls) ? nothing : copy(function_calls)
                        put!(channel, (; chunk..., full_text, function_calls=calls))
                    end
                end
            end
        end
    catch e
        put!(channel, _error_chunk(e))
    finally
        close(channel)
    end

    return channel
end

function generate_content_stream(api_key::String, model_name::String, contents; kwargs...)
    return generate_content_stream(
        GoogleProvider(; api_key), model_name, contents; kwargs...
    )
end

function generate_content_stream(model_name::String, contents; kwargs...)
    return generate_content_stream(GoogleProvider(), model_name, contents; kwargs...)
end

"""
    count_tokens([provider | api_key], model_name, contents; config=GenerateContentConfig()) -> Int

Count the tokens `contents` (anything [`generate_content`](@ref) accepts) would use with
`model_name`. The system instruction, tools and other request fields in `config` are counted
too.
"""
function count_tokens(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents;
    config::GenerateContentConfig=GenerateContentConfig(),
)
    model = _model_path(model_name)
    request = _build_request_body(_to_contents(contents), config)
    request["model"] = model
    body = Dict("generateContentRequest" => request)
    result = _request_json(
        provider, "$model:countTokens", :POST, body; config.http_options...
    )
    return get(result, :totalTokens, 0)
end

function count_tokens(api_key::String, model_name::String, contents; kwargs...)
    return count_tokens(GoogleProvider(; api_key), model_name, contents; kwargs...)
end

function count_tokens(model_name::String, contents; kwargs...)
    return count_tokens(GoogleProvider(), model_name, contents; kwargs...)
end
