# Live / realtime API: bidirectional streaming over WebSocket
# (`client.aio.live.connect()` equivalent in the Python SDK).

const LIVE_WS_PATH = "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
const LIVE_SETUP_TIMEOUT = 30.0 # seconds

"""
    LiveSession

A bidirectional streaming session with the Gemini Live API.
Created via [`connect_live`](@ref); events arrive on `session.events`.
"""
mutable struct LiveSession
    provider::AbstractGoogleProvider
    model::String
    config::GenerateContentConfig
    functions::Dict{String,Function}
    out_channel::Channel{String} # outgoing serialized frames
    events::Channel{Dict{Symbol,Any}} # incoming parsed events
    ws_ref::Base.RefValue{Any} # underlying HTTP.WebSockets connection
    setup_complete::Bool
    closed::Bool
end

"""
    _live_websocket_url(provider)

Build the WebSocket URL for the BidiGenerateContent endpoint.
"""
function _live_websocket_url(provider::AbstractGoogleProvider)
    scheme = startswith(provider.base_url, "http://") ? "ws" : "wss"
    host_port = replace(provider.base_url, r"^https?://" => "", r"/$" => "")
    version = provider.api_version == "v1alpha" ? "v1alpha" : "v1beta"
    path = replace(LIVE_WS_PATH, "v1beta." => "$version.")
    return "$scheme://$host_port/ws/$path?key=$(provider.api_key)"
end

function string_key_dict(d::AbstractDict)
    return Dict{String,Any}(
        string(k) => (v isa AbstractDict ? string_key_dict(v) : v) for (k, v) in pairs(d)
    )
end

"""
    _build_live_tools(config::GenerateContentConfig) -> Vector{Any}

Assemble the `tools` array for a Live setup message from native tool dicts and
function declarations, mirroring `_build_request_body` in generate.jl.
"""
function _build_live_tools(config::GenerateContentConfig)
    tools = Any[]
    function_declarations = Any[]

    if config.tools !== nothing
        for tool in config.tools
            if isa(tool, Function)
                try
                    decl = FunctionDeclaration(tool)
                    push!(function_declarations, to_api_function_declaration(decl))
                catch e
                    @warn "Failed to convert function $(nameof(tool)) to declaration: $e"
                end
            elseif isa(tool, AbstractDict)
                push!(tools, string_key_dict(tool))
            else
                @warn "Ignoring unsupported live tool type: $(typeof(tool))"
            end
        end
    end

    if config.function_declarations !== nothing
        for fd in config.function_declarations
            push!(function_declarations, to_api_function_declaration(fd))
        end
    end

    if !isempty(function_declarations)
        push!(tools, Dict{String,Any}("functionDeclarations" => function_declarations))
    end
    return tools
end

"""
    _build_live_setup(model, config::GenerateContentConfig) -> Dict{String,Any}

Build the `setup` message sent immediately after the WebSocket opens.
"""
function _build_live_setup(model::String, config::GenerateContentConfig)
    setup = Dict{String,Any}("model" => "models/$model")

    generation_config = Dict{String,Any}()
    config.response_modalities !== nothing &&
        (generation_config["responseModalities"] = config.response_modalities)
    config.temperature !== nothing &&
        (generation_config["temperature"] = config.temperature)
    config.top_p !== nothing && (generation_config["topP"] = config.top_p)
    config.top_k !== nothing && (generation_config["topK"] = config.top_k)
    config.max_output_tokens !== nothing &&
        (generation_config["maxOutputTokens"] = config.max_output_tokens)
    config.media_resolution !== nothing &&
        (generation_config["mediaResolution"] = config.media_resolution)
    config.speech_config !== nothing &&
        (generation_config["speechConfig"] = string_key_dict(config.speech_config))
    config.audio_timestamp !== nothing &&
        (generation_config["audioTimestamp"] = config.audio_timestamp)
    isempty(generation_config) || (setup["generationConfig"] = generation_config)

    if config.system_instruction !== nothing
        setup["systemInstruction"] = _format_system_instruction(config.system_instruction)
    end

    tools = _build_live_tools(config)
    isempty(tools) || (setup["tools"] = tools)

    if config.tool_config !== nothing &&
        config.tool_config.function_calling_config !== nothing
        fc = config.tool_config.function_calling_config
        fc_dict = Dict{String,Any}("mode" => fc.mode)
        fc.allowed_function_names !== nothing &&
            (fc_dict["allowedFunctionNames"] = fc.allowed_function_names)
        setup["toolConfig"] = Dict{String,Any}("functionCallingConfig" => fc_dict)
    end

    if config.cached_content !== nothing
        setup["cachedContent"] = config.cached_content
    end

    return setup
end

_send_frame(session::LiveSession, msg::AbstractDict) =
    put!(session.out_channel, JSON3.write(msg))

"""
    connect_live(provider, model; config=GenerateContentConfig(), functions=Dict{String,Function}()) -> LiveSession
    connect_live(api_key, model; kwargs...) -> LiveSession
    connect_live(model; kwargs...) -> LiveSession

Open a bidirectional streaming session with the Gemini Live API over a
WebSocket (`aio.live.connect()` equivalent). Blocks until the server confirms
with `setupComplete`, then returns a [`LiveSession`](@ref).

Keyword arguments:
- `config::GenerateContentConfig`: e.g. `response_modalities=["AUDIO"]` for voice output,
  plus `speech_config`, `system_instruction`, `tools`, `temperature`, ...
- `functions::Dict{String, <:Function}`: executed automatically when the model issues
  a mid-session tool call.

Use [`send_realtime_input`](@ref), [`send_client_content`](@ref),
[`recv_event`](@ref), [`receive_text`](@ref), and [`close_live!`](@ref).
See the README for full examples.
"""
function connect_live(
    provider::AbstractGoogleProvider,
    model::String;
    config::GenerateContentConfig=GenerateContentConfig(),
    functions::Dict{String,<:Function}=Dict{String,Function}(),
)
    url = _live_websocket_url(provider)
    session = LiveSession(
        provider,
        model,
        config,
        Dict{String,Function}(functions),
        Channel{String}(Inf),
        Channel{Dict{Symbol,Any}}(64),
        Base.RefValue{Any}(nothing),
        false,
        false,
    )

    setup_msg = Dict{String,Any}("setup" => _build_live_setup(model, config))
    setup_done = Base.Event()

    session.ws_ref[] = @async begin
        try
            HTTP.WebSockets.open(url) do ws
                # Writer task: forwards queued frames to the socket.
                writer = @async begin
                    try
                        while true
                            frame = take!(session.out_channel)
                            if frame === "\0__CLOSE__"
                                break
                            end
                            HTTP.WebSockets.send(ws, frame)
                        end
                    catch e
                        e isa InvalidStateException && return # channel closed
                        @debug "Live writer task failed" exception = e
                    end
                end

                _send_frame(session, setup_msg)

                for message in ws
                    event = _parse_live_event(session, String(message))
                    if haskey(event, :setupComplete)
                        session.setup_complete = true
                        notify(setup_done)
                    end
                    put!(session.events, event)
                    (haskey(event, :goAway) || haskey(event, :_closed)) && break
                end
                wait(writer)
            end
        catch e
            if !session.setup_complete
                put!(
                    session.events,
                    Dict{Symbol,Any}(:error => "WebSocket connection failed: $e"),
                )
                notify(setup_done)
            end
            @debug "Live session connection ended" exception = e
        finally
            session.closed = true
            notify(setup_done)
            close(session.out_channel)
            close(session.events)
        end
    end

    # Wait for the server to confirm the session, with a hard timeout.
    timer = Timer(LIVE_SETUP_TIMEOUT) do _
        notify(setup_done)
    end
    try
        wait(setup_done)
    finally
        close(timer)
    end

    if !session.setup_complete
        # Give the error event (if any) a moment to surface.
        if isready(session.events)
            event = take!(session.events)
            haskey(event, :error) && error(event[:error])
        end
        error("Timed out waiting for setupComplete after $LIVE_SETUP_TIMEOUT seconds")
    end

    return session
end

function connect_live(api_key::String, model::String; kwargs...)
    return connect_live(GoogleProvider(; api_key), model; kwargs...)
end
connect_live(model::String; kwargs...) = connect_live(GoogleProvider(), model; kwargs...)

"""
    _parse_live_event(session, frame::String) -> Dict{Symbol,Any}

Parse one incoming JSON frame into a symbol-keyed dict. Executes registered
functions and answers `toolCall` frames automatically.
"""
function _parse_live_event(session::LiveSession, frame::String)
    msg = JSON3.read(frame)
    event = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(msg))

    if haskey(event, :toolCall)
        calls = get(event, :functionCalls, get(event[:toolCall], :functionCalls, []))
        if !isempty(session.functions) && !isempty(calls)
            responses = [
                Dict{String,Any}(
                    "id" => String(get(fc, :id, "")),
                    "name" => String(get(fc, :name, "")),
                    "response" => _execute_live_function(
                        session.functions, String(get(fc, :name, "")), get(fc, :args, Dict())
                    ),
                ) for fc in calls
            ]
            try
                _send_frame(session, Dict("toolResponse" => Dict("functionResponses" => responses)))
            catch e
                @warn "Failed to send toolResponse" exception = e
            end
        end
    end

    if haskey(event, :serverContent)
        sc = event[:serverContent]
        audio = _extract_live_audio(sc)
        audio !== nothing && (event[:audio] = audio)
        interrupted = get(sc, :interrupted, false)
        interrupted && (event[:interrupted] = true)
    end

    return event
end

function _execute_live_function(functions::Dict{String,Function}, name::String, args)
    !haskey(functions, name) &&
        return Dict("error" => "Function $name not found")
    try
        symbol_args = string_to_symbol_keys(args)
        result = functions[name](; symbol_args...)
        result === nothing && return Dict("status" => "completed")
        isa(result, AbstractDict) && return string_key_dict(result)
        isa(result, AbstractVector) && return Dict{String,Any}("value" => result)
        return Dict{String,Any}("value" => result)
    catch e
        return Dict{String,Any}("error" => "Error executing function: $e")
    end
end

# Generic fallback so JSON3.Object tool-call args work with keyword splatting.
function string_to_symbol_keys(dict::AbstractDict)
    return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(dict))
end

"""
    _extract_live_audio(server_content) -> Union{Tuple{String,Vector{UInt8}},Nothing}

Extract inline base64-encoded audio output from a `serverContent` message.
Returns `(mime_type, raw_bytes)`.
"""
function _extract_live_audio(server_content)
    model_turn = get(server_content, :modelTurn, nothing)
    model_turn === nothing && return nothing
    for part in get(model_turn, :parts, [])
        inline_data = get(part, :inlineData, get(part, :inline_data, nothing))
        if inline_data !== nothing
            mime_type = String(get(inline_data, :mimeType, "audio/pcm;rate=24000"))
            data = get(inline_data, :data, "")
            isempty(data) && continue
            return mime_type, Base64.base64decode(data)
        end
    end
    return nothing
end

"""
    send_realtime_input(session; audio=nothing, audio_mime_type="audio/pcm;rate=16000",
                        video=nothing, video_mime_type="image/jpeg", text=nothing,
                        activity_start=false, activity_end=false, audio_stream_end=false)

Stream realtime input to the session (`send_realtime_input` equivalent).
`audio` is raw PCM bytes or a file path; `video` is a JPEG/PNG frame as bytes
or path. `audio_stream_end=true` signals the end of an audio stream.
"""
function send_realtime_input(
    session::LiveSession;
    audio=nothing,
    audio_mime_type::String="audio/pcm;rate=16000",
    video=nothing,
    video_mime_type::String="image/jpeg",
    text::Union{Nothing,String}=nothing,
    activity_start::Bool=false,
    activity_end::Bool=false,
    audio_stream_end::Bool=false,
)
    payload = Dict{String,Any}()

    if audio !== nothing
        data = audio isa AbstractVector{UInt8} ? Base64.base64encode(audio) :
               Base64.base64encode(read(String(audio)))
        payload["mediaChunks"] = [Dict{String,Any}("mimeType" => audio_mime_type, "data" => data)]
    elseif audio_stream_end
        payload["audioStreamEnd"] = true
    end

    if video !== nothing
        data = video isa AbstractVector{UInt8} ? Base64.base64encode(video) :
               Base64.base64encode(read(String(video)))
        chunk = Dict{String,Any}("mimeType" => video_mime_type, "data" => data)
        if haskey(payload, "mediaChunks")
            push!(payload["mediaChunks"], chunk)
        else
            payload["mediaChunks"] = [chunk]
        end
    end

    if text !== nothing
        payload["text"] = text
    end

    if activity_start
        payload["activityStart"] = Dict{String,Any}()
    end
    if activity_end
        payload["activityEnd"] = Dict{String,Any}()
    end

    isempty(payload) &&
        throw(ArgumentError("send_realtime_input requires at least one input"))

    _send_frame(session, Dict{String,Any}("realtimeInput" => payload))
    return session
end

"""
    send_client_content(session; turns="", role="user", turn_complete=true)

Send non-realtime content (text turns) into the conversation context;
`turns` is a prompt string or a vector of part dicts.
"""
function send_client_content(
    session::LiveSession;
    turns::Union{String,AbstractVector}="",
    role::String="user",
    turn_complete::Bool=true,
)
    parts = if turns isa String
        [Dict{String,Any}("text" => turns)]
    else
        [part isa AbstractDict ? string_key_dict(part) : part for part in turns]
    end
    msg = Dict{String,Any}(
        "clientContent" =>
            Dict{String,Any}("turns" => [Dict{String,Any}("role" => role, "parts" => parts)], "turnComplete" => turn_complete),
    )
    _send_frame(session, msg)
    return session
end

"""
    send_tool_response(session, responses)

Send results for function calls requested by the model. Only needed when no
matching `functions` were registered at `connect_live` time.
"""
function send_tool_response(session::LiveSession, responses::AbstractVector)
    normalized = [
        r isa AbstractDict ? string_key_dict(r) : throw(ArgumentError("responses must contain dicts")) for r in responses
    ]
    _send_frame(session, Dict{String,Any}("toolResponse" => Dict{String,Any}("functionResponses" => normalized)))
    return session
end

send_tool_response(session::LiveSession, response::AbstractDict) =
    send_tool_response(session, [response])

"""
    recv_event(session) -> Dict{Symbol,Any}

Wait for and return the next event from the model (symbol-keyed dict mirroring
the raw API message, e.g. `setupComplete`, `serverContent`, `toolCall`,
`goAway`; `:audio => (mime, bytes)` is attached when audio was generated).
"""
recv_event(session::LiveSession) = take!(session.events)

"""
    receive_text(session; include_interrupted=true) -> String

Collect events until the current model turn completes; return accumulated text.
"""
function receive_text(session::LiveSession; include_interrupted::Bool=true)
    pieces = String[]
    while true
        event = recv_event(session)
        if haskey(event, :error)
            error(event[:error])
        end
        if haskey(event, :serverContent)
            sc = event[:serverContent]
            model_turn = get(sc, :modelTurn, nothing)
            if model_turn !== nothing
                for part in get(model_turn, :parts, [])
                    text = get(part, :text, nothing)
                    text !== nothing && append!(pieces, [String(text)])
                end
            end
            if get(sc, :interrupted, false) && include_interrupted
                break
            end
            get(sc, :turnComplete, false) && break
            get(sc, :generationComplete, false) && break
        elseif haskey(event, :goAway)
            break
        end
    end
    return join(pieces)
end

"""
    close_live!(session)

Close the WebSocket and tear down background tasks. Idempotent.
"""
function close_live!(session::LiveSession)
    session.closed && return session
    session.closed = true
    try
        put!(session.out_channel, "\0__CLOSE__")
    catch
        # channel already closed; connection died on its own
    end
    return session
end

Base.close(session::LiveSession) = close_live!(session)
Base.isopen(session::LiveSession) = !session.closed && isopen(session.events)

function Base.show(io::IO, ::MIME"text/plain", session::LiveSession)
    println(io, "LiveSession:")
    println(io, "  model:     \"$(session.model)\"")
    println(io, "  open:      $(isopen(session))")
    println(io, "  modality:  $(something(session.config.response_modalities, ["TEXT"]))")
    return print(io, "  functions: $(length(session.functions)) registered")
end
