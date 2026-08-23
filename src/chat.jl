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
- `functions::Dict{String, <:Function}`: Functions the model may call; when non-empty,
  function calls returned by the model are executed automatically and their results
  fed back until the model produces a final answer.
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
    println(io, "  turns:   $(count(entry -> entry[:role] == "user", chat.history))")
    return print(io, "  history: $(length(chat.history)) entries")
end

function _user_message(
    prompt::String; image_path::String="", images::AbstractVector=NamedTuple[]
)
    parts = Dict{String,Any}[Dict("text" => prompt)]
    append!(parts, _image_inline_parts(; image_path, images))
    return Dict(:role => "user", :parts => parts)
end

function _serialize_function_result(result)
    result === nothing && return Dict("status" => "completed")
    isa(result, AbstractDict) && return result
    isa(result, AbstractVector) && return Dict("value" => result)
    return Dict("value" => result)
end

function _execute_function(functions::Dict{String,Function}, fc::FunctionCall)
    if !haskey(functions, fc.name)
        return Dict("error" => "Function $(fc.name) not found")
    end
    try
        symbol_args = string_to_symbol_keys(fc.args)
        return _serialize_function_result(functions[fc.name](; symbol_args...))
    catch e
        return Dict("error" => "Error executing function: $e")
    end
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

function _record_function_exchange!(
    conversation::Vector{<:Dict},
    functions::Dict{String,Function},
    model_parts::Vector{<:Dict},
    function_calls::Vector,
)
    push!(conversation, Dict(:role => "model", :parts => model_parts))

    response_parts = [
        Dict(
            :functionResponse =>
                Dict(:name => fc.name, :response => _execute_function(functions, fc)),
        ) for fc in function_calls
    ]
    push!(conversation, Dict(:role => "user", :parts => response_parts))
    return conversation
end

"""
    send_message(chat, prompt; image_path="", images=NamedTuple[], config=nothing) -> NamedTuple

Send a message in the chat session, appending both the user message and the model
response to the session history. When the chat was created with `functions`,
returned function calls are executed automatically and their results sent back to
the model until a final answer is produced.

# Arguments
- `prompt::String`: The user message.

# Keyword Arguments
- `image_path::String`: Optional path to an image to attach.
- `images::AbstractVector`: Optional additional images (NamedTuples with `path` or `data`).
- `config::Union{Nothing,GenerateContentConfig}`: Per-message override of the chat config.

# Returns
- The final parsed response, same shape as `generate_content`.
"""
function send_message(
    chat::Chat,
    prompt::String;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::Union{Nothing,GenerateContentConfig}=nothing,
)
    active_config = something(config, chat.config)
    user_message = _user_message(prompt; image_path, images)
    conversation = vcat(chat.history, [user_message])

    response = generate_content(
        chat.provider, chat.model, conversation; config=active_config
    )

    for _ in 2:MAX_FUNCTION_CALL_ROUNDS
        (response.function_calls === nothing || isempty(chat.functions)) && break
        model_parts = _content_parts(response)
        model_parts === nothing && break
        _record_function_exchange!(
            conversation, chat.functions, model_parts, response.function_calls
        )
        response = generate_content(
            chat.provider, chat.model, conversation; config=active_config
        )
    end

    if response.function_calls !== nothing && !isempty(chat.functions)
        @warn "Maximum automatic function calling rounds reached"
    end

    append!(chat.history, [user_message])
    model_parts = _content_parts(response)
    if model_parts !== nothing
        append!(chat.history, [Dict(:role => "model", :parts => model_parts)])
    end

    return response
end

"""
    send_message_stream(chat, prompt; image_path="", images=NamedTuple[], config=nothing) -> Channel

Streaming variant of `send_message`. Yields the same named tuples as
`generate_content_stream`, including cumulative `full_text`. If the chat was
created with `functions`, streamed function-call rounds are handled automatically
and their chunks yielded too; history is updated once everything completes.
"""
function send_message_stream(
    chat::Chat,
    prompt::String;
    image_path::String="",
    images::AbstractVector=NamedTuple[],
    config::Union{Nothing,GenerateContentConfig}=nothing,
)
    active_config = something(config, chat.config)
    user_message = _user_message(prompt; image_path, images)
    conversation = vcat(chat.history, [user_message])

    result_channel = Channel{NamedTuple}(32)

    @async begin
        try
            rounds = 0
            aborted = false
            final_chunk = NamedTuple()
            while true
                full_text = ""
                function_calls = nothing
                for chunk in generate_content_stream(
                    chat.provider, chat.model, conversation; config=active_config
                )
                    if haskey(chunk, :error)
                        put!(result_channel, chunk)
                        aborted = true
                        break
                    end
                    haskey(chunk, :full_text) || continue
                    full_text = chunk.full_text
                    chunk.function_calls !== nothing &&
                        (function_calls = chunk.function_calls)
                    final_chunk = chunk
                    put!(result_channel, chunk)
                    chunk.finish_reason !== nothing && break
                end
                aborted && break
                rounds += 1

                if function_calls === nothing ||
                    isempty(chat.functions) ||
                    rounds >= MAX_FUNCTION_CALL_ROUNDS
                    push!(chat.history, user_message)
                    model_parts = _content_parts(final_chunk)
                    if model_parts !== nothing
                        push!(chat.history, Dict(:role => "model", :parts => model_parts))
                    else
                        push!(
                            chat.history,
                            Dict(
                                :role => "model",
                                :parts => [Dict(:text => String(full_text))],
                            ),
                        )
                    end
                    break
                end

                model_parts = _content_parts(final_chunk)
                model_parts === nothing && break
                _record_function_exchange!(
                    conversation, chat.functions, model_parts, function_calls
                )
            end
        catch e
            put!(result_channel, (error=e, text="", finish_reason="ERROR", is_final=true))
        finally
            close(result_channel)
        end
    end

    return result_channel
end

"""
    get_history(chat; curated=false) -> Vector{Dict{Symbol,Any}}

Return the conversation history of the chat session.

With `curated=true`, intermediate function-call exchanges are filtered out,
matching the behavior of `chats[].get_history(curated=True)` in the Python SDK.
"""
function get_history(chat::Chat; curated::Bool=false)
    curated || return copy(chat.history)
    return [
        entry for entry in chat.history if !any(
            part -> haskey(part, :functionCall) || haskey(part, :functionResponse),
            entry[:parts],
        )
    ]
end

"""
    clear_history!(chat)

Remove all messages from the chat session history.
"""
function clear_history!(chat::Chat)
    empty!(chat.history)
    return chat
end
