# Live / realtime API: bidirectional streaming over WebSocket
# (`client.aio.live.connect()` equivalent in the Python SDK).

const LIVE_WS_PATH = "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
const LIVE_SETUP_TIMEOUT = 30.0 # seconds

"""
    LiveSession

A bidirectional streaming session with the Gemini Live API.
Created via [`connect_live`](@ref); events arrive on `session.events`. With session
resumption on, `session.resumption_handle` holds the latest handle for resuming later.
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
    resumption_handle::Union{Nothing,String}
end

function LiveSession(provider, model, config, functions)
    return LiveSession(
        provider,
        model,
        config,
        Dict{String,Function}(functions),
        Channel{String}(Inf),
        Channel{Dict{Symbol,Any}}(64),
        Base.RefValue{Any}(nothing),
        false,
        false,
        nothing,
    )
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
    return "$scheme://$host_port/ws/$path"
end

_live_option(enabled::Bool) = enabled ? Dict{String,Any}() : nothing
_live_option(::Nothing) = nothing
_live_option(options::AbstractDict) = options

_session_resumption(handle::AbstractString) = Dict{String,Any}("handle" => handle)
_session_resumption(enabled) = _live_option(enabled)

function _context_window_compression(enabled::Bool)
    return enabled ? Dict{String,Any}("slidingWindow" => Dict{String,Any}()) : nothing
end
_context_window_compression(options) = _live_option(options)

"""
    _build_live_setup(model, config::GenerateContentConfig; ...) -> Dict{String,Any}

Build the `setup` message sent immediately after the WebSocket opens.
"""
function _build_live_setup(
    model::String,
    config::GenerateContentConfig;
    input_audio_transcription::Bool=false,
    output_audio_transcription::Bool=false,
    session_resumption=nothing,
    context_window_compression=nothing,
    realtime_input_config::Union{Nothing,AbstractDict}=nothing,
)
    setup = _build_request_body(nothing, config)
    delete!(setup, "serviceTier")
    setup["model"] = _model_path(model)
    return merge!(
        setup,
        _compact(
            "inputAudioTranscription" => _live_option(input_audio_transcription),
            "outputAudioTranscription" => _live_option(output_audio_transcription),
            "sessionResumption" => _session_resumption(session_resumption),
            "contextWindowCompression" =>
                _context_window_compression(context_window_compression),
            "realtimeInputConfig" => realtime_input_config,
        ),
    )
end

function _send_frame(session::LiveSession, msg::AbstractDict)
    return put!(session.out_channel, JSON3.write(msg))
end

"""
    connect_live(provider, model; config=GenerateContentConfig(), functions=Dict{String,Function}(), setup_options...) -> LiveSession
    connect_live(api_key, model; kwargs...) -> LiveSession
    connect_live(model; kwargs...) -> LiveSession

Open a bidirectional streaming session with the Gemini Live API over a
WebSocket (`aio.live.connect()` equivalent). Blocks until the server confirms
with `setupComplete`, then returns a [`LiveSession`](@ref).

Keyword arguments:
- `config::GenerateContentConfig`: e.g. `response_modalities=["AUDIO"]` for voice output,
  plus `speech_config`, `system_instruction`, `tools`, `thinking_config`, `temperature`, ...
- `functions::Dict{String, <:Function}`: executed automatically when the model issues
  a mid-session tool call.
- `input_audio_transcription::Bool=false`: Transcribe the audio you send; transcripts arrive
  as `:input_transcription` on events.
- `output_audio_transcription::Bool=false`: Transcribe the model's audio; transcripts arrive
  as `:output_transcription` on events.
- `session_resumption=nothing`: `true` to make the session resumable, or a handle (from
  `session.resumption_handle` of an earlier session) to resume that session.
- `context_window_compression=nothing`: `true` for sliding-window compression with default
  settings, or a dict such as
  `Dict(:triggerTokens => 25_000, :slidingWindow => Dict(:targetTokens => 12_000))`.
- `realtime_input_config=nothing`: Voice activity detection and turn settings, e.g.
  `Dict(:automaticActivityDetection => Dict(:disabled => true))` to signal activity yourself
  with `send_realtime_input(session; activity_start=true)`.

Use [`send_realtime_input`](@ref), [`send_client_content`](@ref),
[`recv_event`](@ref), [`receive_text`](@ref), and [`close_live!`](@ref).
See the README for full examples.
"""
function connect_live(
    provider::AbstractGoogleProvider,
    model::String;
    config::GenerateContentConfig=GenerateContentConfig(),
    functions::Dict{String,<:Function}=Dict{String,Function}(),
    setup_options...,
)
    url = _live_websocket_url(provider)
    session = LiveSession(provider, model, config, functions)
    setup_msg = Dict{String,Any}(
        "setup" => _build_live_setup(model, config; setup_options...)
    )
    setup_done = Base.Event()

    session.ws_ref[] = @async begin
        try
            HTTP.WebSockets.open(url; headers=_headers(provider)) do ws
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
                        e isa InvalidStateException && return nothing # channel closed
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
                put!(session.events, Dict{Symbol,Any}(:error => _live_error(e)))
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
        return notify(setup_done)
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
            haskey(event, :error) && throw(event[:error])
        end
        error("Timed out waiting for setupComplete after $LIVE_SETUP_TIMEOUT seconds")
    end

    return session
end

# Turn a failed handshake or an error close frame into a `GeminiAPIError`.
function _live_error(e)
    e isa HTTP.RequestError && (e = e.error)
    e isa HTTP.StatusError && return GeminiAPIError(e.response)
    if e isa HTTP.WebSockets.WebSocketError && e.message isa HTTP.WebSockets.CloseFrameBody
        return GeminiAPIError(e.message.status, "", e.message.message, nothing, nothing)
    end
    return e
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
                    "response" => _call_function(
                        session.functions,
                        String(get(fc, :name, "")),
                        get(fc, :args, Dict()),
                    ),
                ) for fc in calls
            ]
            try
                _send_frame(
                    session, Dict("toolResponse" => Dict("functionResponses" => responses))
                )
            catch e
                @warn "Failed to send toolResponse" exception = e
            end
        end
    end

    if haskey(event, :serverContent)
        sc = event[:serverContent]
        audio = _extract_live_audio(sc)
        audio !== nothing && (event[:audio] = audio)
        get(sc, :interrupted, false) && (event[:interrupted] = true)
        for (field, key) in (
            :inputTranscription => :input_transcription,
            :outputTranscription => :output_transcription,
        )
            transcription = get(sc, field, nothing)
            transcription === nothing ||
                (event[key] = String(get(transcription, :text, "")))
        end
    end

    update = get(event, :sessionResumptionUpdate, nothing)
    if update !== nothing && get(update, :resumable, false)
        handle = get(update, :newHandle, "")
        isempty(handle) || (session.resumption_handle = String(handle))
    end

    return event
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

function _media_chunk(media, mime_type)
    bytes = media isa AbstractVector{UInt8} ? media : read(String(media))
    return Dict{String,Any}("mimeType" => mime_type, "data" => base64encode(bytes))
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
    chunks = [
        _media_chunk(x, mime) for
        (x, mime) in ((audio, audio_mime_type), (video, video_mime_type)) if x !== nothing
    ]
    isempty(chunks) || (payload["mediaChunks"] = chunks)
    audio === nothing && audio_stream_end && (payload["audioStreamEnd"] = true)

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
        "clientContent" => Dict{String,Any}(
            "turns" => [Dict{String,Any}("role" => role, "parts" => parts)],
            "turnComplete" => turn_complete,
        ),
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
        if r isa AbstractDict
            string_key_dict(r)
        else
            throw(ArgumentError("responses must contain dicts"))
        end for r in responses
    ]
    _send_frame(
        session,
        Dict{String,Any}(
            "toolResponse" => Dict{String,Any}("functionResponses" => normalized)
        ),
    )
    return session
end

function send_tool_response(session::LiveSession, response::AbstractDict)
    return send_tool_response(session, [response])
end

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
        haskey(event, :error) && throw(event[:error])
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
