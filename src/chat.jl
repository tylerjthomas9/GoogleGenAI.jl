"""
    Chat

A chat session with persistent history, mirroring `client.chats.create()` from the
Python SDK. Holds a provider, model, generation config, optional registered Julia
functions for automatic function calling, and the running conversation history.
"""
mutable struct Chat
    provider::AbstractGoogleProvider
    model::String
    config::GenerateContentConfig
    functions::Dict{String,Function}
    history::Vector{Dict{Symbol,Any}}
end

const MAX_FUNCTION_CALL_ROUNDS = 5

"""
    create_chat(provider, model; config=GenerateContentConfig(), functions=Dict{String,Function}(), history=Dict{Symbol,Any}[]) -> Chat
    create_chat(api_key, model; kwargs...) -> Chat
    create_chat(model; kwargs...) -> Chat

Create a persistent chat session. Equivalent to `client.chats.create(...)` in Python.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string.
- `model::String`: The model to use for the chat.

# Keyword Arguments
- `config::GenerateContentConfig`: Configuration applied to every message.
- `functions::Dict{String, <:Function}`: Implementations, by name, of the functions declared
  in `config`; when non-empty, function calls returned by the model are executed
  automatically and their results fed back until the model produces a final answer.
- `history::Vector{Dict{Symbol,Any}}`: Optional initial conversation history.

# Examples
```julia
chat = create_chat(provider, "gemini-3.7-flash")
response = send_message(chat, "Hello!")
response = send_message(chat, "What did I just say?")
```
"""
function create_chat(
    provider::AbstractGoogleProvider,
    model::String;
    config::GenerateContentConfig=GenerateContentConfig(),
    functions::Dict{String,<:Function}=Dict{String,Function}(),
    history::Vector{Dict{Symbol,Any}}=Dict{Symbol,Any}[],
)
    return Chat(provider, model, config, Dict{String,Function}(functions), history)
end

function create_chat(api_key::String, model::String; kwargs...)
    return create_chat(GoogleProvider(; api_key), model; kwargs...)
end
create_chat(model::String; kwargs...) = create_chat(GoogleProvider(), model; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", chat::Chat)
    println(io, "Chat:")
    println(io, "  model:   \"$(chat.model)\"")
    println(io, "  turns:   $(count(entry -> _key(entry, :role) == "user", chat.history))")
    return print(io, "  history: $(length(chat.history)) entries")
end

function _content_parts(response)::Union{Nothing,Vector{Dict{Symbol,Any}}}
    candidates = get(response, :candidates, nothing)
    (candidates === nothing || isempty(candidates)) && return nothing
    content = get(candidates[1], :content, nothing)
    (content === nothing || !haskey(content, :parts)) && return nothing
    return [
        Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(part)) for part in content.parts
    ]
end

function _model_turn(parts)
    return Dict{Symbol,Any}(
        :role => "model", :parts => something(parts, Dict{Symbol,Any}[])
    )
end

function _is_text_part(part)
    return haskey(part, :text) && keys(part) ⊆ (:text, :thought, :thoughtSignature)
end

function _append_stream_parts!(parts::Vector{Dict{Symbol,Any}}, chunk)
    for part in something(_content_parts(chunk), ())
        if !_is_text_part(part)
            push!(parts, part)
            continue
        end
        prev = isempty(parts) ? nothing : parts[end]
        if prev !== nothing &&
            _is_text_part(prev) &&
            !haskey(prev, :thoughtSignature) &&
            get(prev, :thought, false) == get(part, :thought, false)
            prev[:text] = string(prev[:text], part[:text])
            haskey(part, :thoughtSignature) &&
                (prev[:thoughtSignature] = part[:thoughtSignature])
        elseif !isempty(part[:text]) || haskey(part, :thoughtSignature)
            push!(parts, part)
        end
    end
    return parts
end

function _record_function_exchange!(
    conversation::Vector,
    functions::Dict{String,Function},
    model_parts,
    function_calls::Vector,
)
    push!(conversation, _model_turn(model_parts))
    response_parts = [
        Dict(
            :functionResponse => Dict(
                :name => fc.name,
                :response => _call_function(functions, fc.name, fc.args),
            ),
        ) for fc in function_calls
    ]
    push!(conversation, Dict{Symbol,Any}(:role => "user", :parts => response_parts))
    return conversation
end

function _is_valid_turn(turn)
    parts = _key(turn, :parts)
    return parts !== nothing && !isempty(parts) && all(!isempty, parts)
end

# Drops model replies that came back empty, along with the user turn that prompted them.
function _curated_history(history)
    curated = Dict{Symbol,Any}[]
    i = 1
    while i <= length(history)
        if _key(history[i], :role) != "model"
            push!(curated, history[i])
            i += 1
            continue
        end
        j = i
        while j <= length(history) && _key(history[j], :role) == "model"
            j += 1
        end
        replies = history[i:(j - 1)]
        if all(_is_valid_turn, replies)
            append!(curated, replies)
        elseif !isempty(curated)
            pop!(curated)
        end
        i = j
    end
    return curated
end

function _user_turns(message, image_path, images)
    return Dict{Symbol,Any}[_with_media(message, image_path, images)...]
end

"""
    send_message(chat, message; image_path="", images=NamedTuple[], config=nothing) -> NamedTuple

Send a message in a chat session and return the model's response.

`message` is a string or anything else [`generate_content`](@ref) accepts as `contents`.
When the chat was created with `functions`, function calls are executed automatically and
their results sent back until the model answers. Every turn, including function calls,
function results and replies that came back empty, is added to the history; see
[`get_history`](@ref).

# Keyword Arguments
- `image_path::String`: Optional path to an image to attach.
- `images::AbstractVector`: Optional additional media (NamedTuples with `path` or `data`).
- `config::Union{Nothing,GenerateContentConfig}`: Per-message override of the chat config.

# Returns
- The final parsed response, same shape as `generate_content`.
"""
function send_message(
    chat::Chat,
    message;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::Union{Nothing,GenerateContentConfig}=nothing,
)
    active_config = something(config, chat.config)
    turns = _user_turns(message, image_path, images)
    ask() = generate_content(
        chat.provider,
        chat.model,
        [_curated_history(chat.history); turns];
        config=active_config,
    )

    response = ask()
    for _ in 2:MAX_FUNCTION_CALL_ROUNDS
        (response.function_calls === nothing || isempty(chat.functions)) && break
        _record_function_exchange!(
            turns, chat.functions, _content_parts(response), response.function_calls
        )
        response = ask()
    end
    if response.function_calls !== nothing && !isempty(chat.functions)
        @warn "Maximum automatic function calling rounds reached"
    end

    push!(turns, _model_turn(_content_parts(response)))
    append!(chat.history, turns)
    return response
end

"""
    send_message_stream(chat, message; image_path="", images=NamedTuple[], config=nothing) -> Channel

Streaming version of [`send_message`](@ref): yields the same chunks as
[`generate_content_stream`](@ref). With registered `functions`, follow-up rounds are streamed
and their chunks yielded too; the history is updated once the reply is complete, and left
unchanged if the request fails.
"""
function send_message_stream(
    chat::Chat,
    message;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::Union{Nothing,GenerateContentConfig}=nothing,
)
    active_config = something(config, chat.config)
    turns = _user_turns(message, image_path, images)
    channel = Channel{NamedTuple}(32)

    @async try
        for round in 1:MAX_FUNCTION_CALL_ROUNDS
            parts = Dict{Symbol,Any}[]
            function_calls = nothing
            conversation = [_curated_history(chat.history); turns]
            for chunk in generate_content_stream(
                chat.provider, chat.model, conversation; config=active_config
            )
                put!(channel, chunk)
                haskey(chunk, :error) && return nothing
                _append_stream_parts!(parts, chunk)
                function_calls = chunk.function_calls
            end
            if function_calls === nothing ||
                isempty(chat.functions) ||
                round == MAX_FUNCTION_CALL_ROUNDS
                push!(turns, _model_turn(parts))
                append!(chat.history, turns)
                return nothing
            end
            _record_function_exchange!(turns, chat.functions, parts, function_calls)
        end
    catch e
        put!(channel, _error_chunk(e))
    finally
        close(channel)
    end

    return channel
end

"""
    get_history(chat; curated=false) -> Vector{Dict{Symbol,Any}}

Return the conversation history of the chat session.

The full history has every turn, including function calls, function results and model
replies that came back empty (e.g. when blocked). With `curated=true`, empty replies and the
user turns that prompted them are left out; this is the history sent with each new message,
matching `chat.get_history(curated=True)` in the Python SDK.
"""
function get_history(chat::Chat; curated::Bool=false)
    return curated ? _curated_history(chat.history) : copy(chat.history)
end

"""
    clear_history!(chat)

Remove all messages from the chat session history.
"""
function clear_history!(chat::Chat)
    empty!(chat.history)
    return chat
end
