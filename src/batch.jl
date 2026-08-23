"""
    BatchRequest(prompt::String; key::String="", config::GenerateContentConfig=GenerateContentConfig())

A single request inside an inline batch job. `key` is an optional user-defined
identifier used to match each response to its request in the batch output;
if empty, a default key `"request-N"` is assigned when the job is created.
"""
Base.@kwdef struct BatchRequest
    prompt::String
    key::String = ""
    config::GenerateContentConfig = GenerateContentConfig()
end

BatchRequest(prompt::AbstractString; kwargs...) = BatchRequest(; prompt=String(prompt), kwargs...)

const COMPLETED_BATCH_STATES = (
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
)

_is_completed_state(state::AbstractString) = state ∈ COMPLETED_BATCH_STATES

function _batch_request_payload(req::BatchRequest)
    request_body = Dict{String,Any}(
        "contents" => [
            Dict("role" => "user", "parts" => [Dict("text" => req.prompt)])
        ],
        "generationConfig" => GoogleGenAI._build_generation_config(req.config),
    )
    if req.config.system_instruction !== nothing
        request_body["systemInstruction"] =
            GoogleGenAI._format_system_instruction(req.config.system_instruction)
    end
    payload = Dict{String,Any}("request" => request_body)
    if !isempty(req.key)
        payload["metadata"] = Dict("key" => req.key)
    end
    return payload
end

function _build_inline_requests(requests::Vector{<:BatchRequest})
    payloads = Vector{Dict{String,Any}}(undef, length(requests))
    for (i, req) in enumerate(requests)
        payload = _batch_request_payload(req)
        key = isempty(req.key) ? "request-$i" : req.key
        payload["metadata"] = Dict("key" => key)
        payloads[i] = payload
    end
    return payloads
end

function _build_batch_request_body(; requests::Union{Nothing,Vector{BatchRequest}}=nothing, input_file::Union{Nothing,String}=nothing, display_name::String="")
    if (requests === nothing) == (input_file === nothing)
        throw(ArgumentError("exactly one of `requests` or `input_file` must be provided"))
    end

    input_config = if requests !== nothing
        Dict("requests" => Dict("requests" => _build_inline_requests(requests)))
    else
        Dict("requests" => Dict("file_name" => input_file))
    end

    batch = Dict{String,Any}("input_config" => input_config)
    isempty(display_name) || (batch["display_name"] = display_name)
    return Dict{String,Any}("batch" => batch)
end

"""
    create_batch_job(
        api_thing, model_name::String;
        requests::Union{Nothing,Vector{BatchRequest}}=nothing,
        input_file::Union{Nothing,String}=nothing,
        display_name::String="",
        http_kwargs...,
    ) -> JSON3.Object

Creates an asynchronous batch generation job (`50%` cost of interactive calls,
target turnaround time of 24 hours). Pass either a vector of
[`BatchRequest`](@ref) (inline requests; total request size must stay under
20MB), or the name of a JSONL file previously uploaded through the File API as
`input_file`. Each line of the input file must be a JSON object of the form
`{"key": "...", "request": {...GenerateContentRequest...}}`.

Returns the created batch operation object; use `obj[:name]` (e.g.
`"batches/123456"`) with [`get_batch_job`](@ref), [`poll_batch_job`](@ref),
[`cancel_batch_job`](@ref), and [`delete_batch_job`](@ref).

# Example
```julia
job = create_batch_job("gemini-2.5-flash"; requests=[
    BatchRequest("Tell me a one-sentence joke."; key="request-1"),
    BatchRequest("Why is the sky blue?"; key="request-2"),
])
```
"""
function create_batch_job(
    provider::AbstractGoogleProvider,
    model_name::String;
    requests::Union{Nothing,Vector{BatchRequest}}=nothing,
    input_file::Union{Nothing,String}=nothing,
    display_name::String="",
    http_kwargs...,
)
    if (requests === nothing) == (input_file === nothing)
        throw(ArgumentError("exactly one of `requests` or `input_file` must be provided"))
    end
    model_name = startswith(model_name, "models/") ? model_name : "models/$model_name"

    body = _build_batch_request_body(; requests=requests, input_file=input_file, display_name=display_name)

    response = _request(
        provider,
        "$model_name:batchGenerateContent",
        :POST,
        body;
        http_kwargs...,
    )
    return JSON3.read(String(response.body))
end

function create_batch_job(
    api_key::String,
    model_name::String;
    kwargs...,
)
    return create_batch_job(GoogleProvider(; api_key=api_key), model_name; kwargs...)
end

function create_batch_job(
    model_name::String;
    kwargs...,
)
    return create_batch_job(GoogleProvider(), model_name; kwargs...)
end

"""
    get_batch_job(api_thing, job_name::String; http_kwargs...) -> JSON3.Object

Returns the batch job (operation) with the given name, e.g. `"batches/123456"`.
Inspect `metadata.state` for the job state (one of `JOB_STATE_PENDING`,
`JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
`JOB_STATE_CANCELLED`, `JOB_STATE_EXPIRED`) and `response` for results once
the job has succeeded.
"""
function get_batch_job(
    provider::AbstractGoogleProvider,
    job_name::String;
    http_kwargs...,
)
    response = _request(provider, job_name, :GET, Dict(); http_kwargs...)
    return JSON3.read(String(response.body))
end

function get_batch_job(api_key::String, job_name::String; http_kwargs...)
    return get_batch_job(GoogleProvider(; api_key=api_key), job_name; http_kwargs...)
end

get_batch_job(job_name::String; http_kwargs...) =
    get_batch_job(GoogleProvider(), job_name; http_kwargs...)

"""
    poll_batch_job(api_thing, job_name::String; interval_s::Real=30, timeout_s::Real=86_400, verbose::Bool=false, http_kwargs...) -> JSON3.Object

Polls [`get_batch_job`](@ref) every `interval_s` seconds until the job reaches
a terminal state (`JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
`JOB_STATE_CANCELLED`, or `JOB_STATE_EXPIRED`) or `timeout_s` elapses, in which
case an `ErrorException` is thrown.
"""
function poll_batch_job(
    provider::AbstractGoogleProvider,
    job_name::String;
    interval_s::Real=30,
    timeout_s::Real=86_400,
    verbose::Bool=false,
    http_kwargs...,
)
    deadline = time() + timeout_s
    while true
        job = get_batch_job(provider, job_name; http_kwargs...)
        state = something(
            _batch_state(job),
            "JOB_STATE_PENDING",
        )
        _is_completed_state(state) && return job
        verbose && @info "Batch job $job_name not finished (state: $state). Waiting $(interval_s)s..."
        time() > deadline && error("Timed out polling batch job $job_name after $timeout_s seconds")
        sleep(interval_s)
    end
end

function poll_batch_job(api_key::String, job_name::String; kwargs...)
    return poll_batch_job(GoogleProvider(; api_key=api_key), job_name; kwargs...)
end

poll_batch_job(job_name::String; kwargs...) =
    poll_batch_job(GoogleProvider(), job_name; kwargs...)

function _batch_state(job::JSON3.Object)
    meta = get(job, :metadata, nothing)
    meta === nothing && return nothing
    return get(meta, :state, nothing)
end

"""
    cancel_batch_job(api_thing, job_name::String; http_kwargs...)

Cancels an ongoing batch job. The job stops processing new requests and its
final state becomes `JOB_STATE_CANCELLED`.
"""
function cancel_batch_job(
    provider::AbstractGoogleProvider,
    job_name::String;
    http_kwargs...,
)
    response = _request(provider, "$job_name:cancel", :POST, Dict(); http_kwargs...)
    return JSON3.read(String(response.body))
end

cancel_batch_job(api_key::String, job_name::String; http_kwargs...) =
    cancel_batch_job(GoogleProvider(; api_key=api_key), job_name; http_kwargs...)

cancel_batch_job(job_name::String; http_kwargs...) =
    cancel_batch_job(GoogleProvider(), job_name; http_kwargs...)

"""
    delete_batch_job(api_thing, job_name::String; http_kwargs...)

Deletes a batch job. The job stops processing new requests and is removed from
the list of batch jobs.
"""
function delete_batch_job(
    provider::AbstractGoogleProvider,
    job_name::String;
    http_kwargs...,
)
    response = _request(provider, "$job_name:delete", :DELETE, Dict(); http_kwargs...)
    return JSON3.read(String(response.body))
end

delete_batch_job(api_key::String, job_name::String; http_kwargs...) =
    delete_batch_job(GoogleProvider(; api_key=api_key), job_name; http_kwargs...)

delete_batch_job(job_name::String; http_kwargs...) =
    delete_batch_job(GoogleProvider(), job_name; http_kwargs...)

"""
    download_batch_output_file(api_thing, file_name::String; http_kwargs...) -> Vector{JSON3.Object}

Downloads and parses the JSONL output file produced by a file-based batch job.
Each parsed line is either a `GenerateContentResponse` or a status object
describing an error for that specific request.
"""
function download_batch_output_file(
    provider::AbstractGoogleProvider,
    file_name::String;
    http_kwargs...,
)
    url = "$(provider.base_url)/download/$(provider.api_version)/$file_name:download?alt=media&key=$(provider.api_key)"
    response = HTTP.get(url; http_kwargs...)
    response.status >= 400 && status_error(response, String(response.body))
    content = String(response.body)
    lines = filter(!isempty, strip.(split(content, '\n')))
    return [JSON3.read(line) for line in lines]
end

download_batch_output_file(file_name::String; http_kwargs...) =
    download_batch_output_file(GoogleProvider(), file_name; http_kwargs...)

download_batch_output_file(api_key::String, file_name::String; http_kwargs...) =
    download_batch_output_file(GoogleProvider(; api_key=api_key), file_name; http_kwargs...)

"""
    extract_batch_text(result)::Vector{Pair{String,String}}

Extracts `(key => generated_text)` pairs from batch job results. Accepts either
the completed job object returned by [`get_batch_job`](@ref) /
[`poll_batch_job`](@ref) (inline responses), or a single parsed line from
[`download_batch_output_file`](@ref).

Failed requests are reported as `key => "ERROR: ..."` entries based on their
status objects or the job's `error` field.
"""
function extract_batch_text(result::JSON3.Object)
    pairs_out = Pair{String,String}[]

    # File-based result line: {"key": ..., "response": {...}} or error status
    if haskey(result, :key) && haskey(result, :response) && result.response !== nothing
        push!(pairs_out, string(result.key) => _generate_content_response_text(result.response))
        return pairs_out
    elseif haskey(result, :key) && haskey(result, :error) && result.error !== nothing
        push!(pairs_out, string(result.key) => "ERROR: $(get(result.error, :message, result.error))")
        return pairs_out
    end

    # Completed job object with inline responses
    response = get(result, :response, nothing)

    # Job-level failure (e.g. JOB_STATE_FAILED with an error field)
    if response === nothing
        err = get(result, :error, nothing)
        if err !== nothing
            push!(pairs_out, "" => "ERROR: $(get(err, :message, err))")
        end
        return pairs_out
    end

    inlined = get(response, :inlinedResponses, nothing)
    if inlined !== nothing
        for inline_response in inlined
            if get(inline_response, :response, nothing) !== nothing
                push!(
                    pairs_out,
                    string(get(get(inline_response, :metadata, Dict()), :key, "")) =>
                        _generate_content_response_text(inline_response.response),
                )
            elseif get(inline_response, :error, nothing) !== nothing
                err = inline_response.error
                push!(
                    pairs_out,
                    string(get(get(inline_response, :metadata, Dict()), :key, "")) =>
                        "ERROR: $(get(err, :message, err))",
                )
            end
        end
        return pairs_out
    end

    # File-based job: point the caller at the output file
    file_name = get(response, :responsesFile, nothing)
    if file_name !== nothing
        throw(ErrorException(
            "This batch job wrote its results to file `$file_name`; call " *
            "`download_batch_output_file(\"$file_name\")` and pass each parsed line to " *
            "`extract_batch_text` individually.",
        ))
    end

    return pairs_out
end

# Best-effort text extraction from a GenerateContentResponse object.
function _generate_content_response_text(response::JSON3.Object)
    texts = String[]
    candidates = get(response, :candidates, nothing)
    candidates === nothing && return ""
    for candidate in candidates
        content = get(candidate, :content, nothing)
        content === nothing && continue
        parts = get(content, :parts, nothing)
        parts === nothing && continue
        for part in parts
            text = get(part, :text, nothing)
            text !== nothing && push!(texts, String(text))
        end
    end
    return join(texts)
end

_generate_content_response_text(other) = ""
