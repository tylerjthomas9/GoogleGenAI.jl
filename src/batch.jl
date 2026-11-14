"""
    BatchRequest(prompt; key::String="", config::GenerateContentConfig=GenerateContentConfig())

A single request inside an inline batch job. `prompt` is anything [`generate_content`](@ref)
accepts as `contents`, and every field of `config` that applies to a request (system
instruction, tools, tool config, safety settings, generation settings, cache) is sent with it.
`key` is an optional identifier used to match each response to its request in the batch
output; if empty, a default key `"request-N"` is assigned when the job is created.
"""
Base.@kwdef struct BatchRequest
    prompt::Any
    key::String = ""
    config::GenerateContentConfig = GenerateContentConfig()
end

BatchRequest(prompt; kwargs...) = BatchRequest(; prompt, kwargs...)

const COMPLETED_BATCH_STATES = (
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"
)

_is_completed_state(state::AbstractString) = state ∈ COMPLETED_BATCH_STATES

function _batch_request_payload(req::BatchRequest, index::Integer=1)
    return Dict{String,Any}(
        "request" => _build_request_body(_to_contents(req.prompt), req.config),
        "metadata" => Dict("key" => isempty(req.key) ? "request-$index" : req.key),
    )
end

function _batch_body(inline, input_file, display_name)
    if (inline === nothing) == (input_file === nothing)
        throw(ArgumentError("exactly one of `requests` or `input_file` must be provided"))
    end
    source = if inline === nothing
        Dict("file_name" => input_file)
    else
        Dict("requests" => Dict("requests" => inline))
    end
    name = isempty(display_name) ? nothing : display_name
    return Dict{String,Any}(
        "batch" => _compact("input_config" => source, "display_name" => name)
    )
end

function _build_batch_request_body(;
    requests::Union{Nothing,Vector{BatchRequest}}=nothing,
    input_file::Union{Nothing,String}=nothing,
    display_name::String="",
)
    inline = if requests === nothing
        nothing
    else
        [_batch_request_payload(r, i) for (i, r) in enumerate(requests)]
    end
    return _batch_body(inline, input_file, display_name)
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
job = create_batch_job("gemini-3.5-flash"; requests=[
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
    body = _build_batch_request_body(; requests, input_file, display_name)
    endpoint = "$(_model_path(model_name)):batchGenerateContent"
    return _request_json(provider, endpoint, :POST, body; http_kwargs...)
end

function create_batch_job(api_key::String, model_name::String; kwargs...)
    return create_batch_job(GoogleProvider(; api_key), model_name; kwargs...)
end

function create_batch_job(model_name::String; kwargs...)
    return create_batch_job(GoogleProvider(), model_name; kwargs...)
end

"""
    create_embeddings_batch_job(
        api_thing, model_name::String;
        contents::Union{Nothing,AbstractVector}=nothing,
        input_file::Union{Nothing,String}=nothing,
        display_name::String="",
        task_type=nothing, title=nothing, output_dimensionality=nothing,
        http_kwargs...,
    ) -> JSON3.Object

Creates an asynchronous batch embedding job. Pass either `contents`, a vector with one entry
per embedding (in any form [`embed_content`](@ref) accepts), or the name of an uploaded JSONL
file as `input_file`. `task_type`, `title` and `output_dimensionality` apply to every inline
request. Track the job with [`get_batch_job`](@ref) and [`poll_batch_job`](@ref).
"""
function create_embeddings_batch_job(
    provider::AbstractGoogleProvider,
    model_name::String;
    contents::Union{Nothing,AbstractVector}=nothing,
    input_file::Union{Nothing,String}=nothing,
    display_name::String="",
    task_type=nothing,
    title=nothing,
    output_dimensionality=nothing,
    http_kwargs...,
)
    model = _model_path(model_name)
    inline = if contents === nothing
        nothing
    else
        [
            Dict(
                "request" =>
                    _embed_request(model, c; task_type, title, output_dimensionality),
                "metadata" => Dict("key" => "request-$i"),
            ) for (i, c) in enumerate(contents)
        ]
    end
    body = _batch_body(inline, input_file, display_name)
    return _request_json(
        provider, "$model:asyncBatchEmbedContent", :POST, body; http_kwargs...
    )
end

function create_embeddings_batch_job(api_key::String, model_name::String; kwargs...)
    return create_embeddings_batch_job(GoogleProvider(; api_key), model_name; kwargs...)
end

function create_embeddings_batch_job(model_name::String; kwargs...)
    return create_embeddings_batch_job(GoogleProvider(), model_name; kwargs...)
end

"""
    list_batch_jobs(api_thing; page_size=nothing, http_kwargs...) -> Vector

Lists all batch jobs (operation objects) in your project, fetching all pages.
"""
function list_batch_jobs(
    provider::AbstractGoogleProvider; page_size=nothing, http_kwargs...
)
    return _paginate(provider, "batches", :operations; page_size, http_kwargs...)
end

function list_batch_jobs(api_key::String; kwargs...)
    return list_batch_jobs(GoogleProvider(; api_key); kwargs...)
end
list_batch_jobs(; kwargs...) = list_batch_jobs(GoogleProvider(); kwargs...)

"""
    get_batch_job(api_thing, job_name::String; http_kwargs...) -> JSON3.Object

Returns the batch job (operation) with the given name, e.g. `"batches/123456"`.
Inspect `metadata.state` for the job state (one of `JOB_STATE_PENDING`,
`JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
`JOB_STATE_CANCELLED`, `JOB_STATE_EXPIRED`) and `response` for results once
the job has succeeded.
"""
function get_batch_job(provider::AbstractGoogleProvider, job_name::String; http_kwargs...)
    return _request_json(provider, job_name, :GET; http_kwargs...)
end

function get_batch_job(api_key::String, job_name::String; http_kwargs...)
    return get_batch_job(GoogleProvider(; api_key), job_name; http_kwargs...)
end

function get_batch_job(job_name::String; http_kwargs...)
    return get_batch_job(GoogleProvider(), job_name; http_kwargs...)
end

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
        state = something(_batch_state(job), "JOB_STATE_PENDING")
        _is_completed_state(state) && return job
        verbose &&
            @info "Batch job $job_name not finished (state: $state). Waiting $(interval_s)s..."
        time() > deadline &&
            error("Timed out polling batch job $job_name after $timeout_s seconds")
        sleep(interval_s)
    end
end

function poll_batch_job(api_key::String, job_name::String; kwargs...)
    return poll_batch_job(GoogleProvider(; api_key), job_name; kwargs...)
end

function poll_batch_job(job_name::String; kwargs...)
    return poll_batch_job(GoogleProvider(), job_name; kwargs...)
end

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
    provider::AbstractGoogleProvider, job_name::String; http_kwargs...
)
    return _request_json(provider, "$job_name:cancel", :POST; http_kwargs...)
end

function cancel_batch_job(api_key::String, job_name::String; http_kwargs...)
    return cancel_batch_job(GoogleProvider(; api_key), job_name; http_kwargs...)
end

function cancel_batch_job(job_name::String; http_kwargs...)
    return cancel_batch_job(GoogleProvider(), job_name; http_kwargs...)
end

"""
    delete_batch_job(api_thing, job_name::String; http_kwargs...)

Deletes a batch job. The job stops processing new requests and is removed from
the list of batch jobs.
"""
function delete_batch_job(
    provider::AbstractGoogleProvider, job_name::String; http_kwargs...
)
    return _request_json(provider, job_name, :DELETE; http_kwargs...)
end

function delete_batch_job(api_key::String, job_name::String; http_kwargs...)
    return delete_batch_job(GoogleProvider(; api_key), job_name; http_kwargs...)
end

function delete_batch_job(job_name::String; http_kwargs...)
    return delete_batch_job(GoogleProvider(), job_name; http_kwargs...)
end

"""
    download_batch_output_file(api_thing, file_name::String; http_kwargs...) -> Vector{JSON3.Object}

Downloads and parses the JSONL output file produced by a file-based batch job.
Each parsed line is either a `GenerateContentResponse` or a status object
describing an error for that specific request.
"""
function download_batch_output_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs...
)
    content = String(download_file(provider, file_name; http_kwargs=(; http_kwargs...)))
    return [
        JSON3.read(line) for line in eachline(IOBuffer(content)) if !isempty(strip(line))
    ]
end

function download_batch_output_file(api_key::String, file_name::String; http_kwargs...)
    return download_batch_output_file(GoogleProvider(; api_key), file_name; http_kwargs...)
end

function download_batch_output_file(file_name::String; http_kwargs...)
    return download_batch_output_file(GoogleProvider(), file_name; http_kwargs...)
end

_error_text(err) = "ERROR: $(get(err, :message, err))"
_result_key(result) = string(get(get(result, :metadata, Dict()), :key, ""))

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
    # File-based result line: {"key": ..., "response": {...}} or {"key": ..., "error": {...}}
    if haskey(result, :key)
        if get(result, :response, nothing) !== nothing
            return [string(result.key) => _generate_content_response_text(result.response)]
        elseif get(result, :error, nothing) !== nothing
            return [string(result.key) => _error_text(result.error)]
        end
    end

    response = get(result, :response, nothing)
    if response === nothing
        err = get(result, :error, nothing)
        return err === nothing ? Pair{String,String}[] : ["" => _error_text(err)]
    end

    inlined = get(response, :inlinedResponses, nothing)
    # The API nests the list: `{"inlinedResponses": {"inlinedResponses": [...]}}`.
    inlined isa AbstractDict && (inlined = get(inlined, :inlinedResponses, ()))
    if inlined !== nothing
        pairs_out = Pair{String,String}[]
        for item in inlined
            if get(item, :response, nothing) !== nothing
                push!(
                    pairs_out,
                    _result_key(item) => _generate_content_response_text(item.response),
                )
            elseif get(item, :error, nothing) !== nothing
                push!(pairs_out, _result_key(item) => _error_text(item.error))
            end
        end
        return pairs_out
    end

    file_name = get(response, :responsesFile, nothing)
    if file_name !== nothing
        throw(
            ErrorException(
                "This batch job wrote its results to file `$file_name`; call " *
                "`download_batch_output_file(\"$file_name\")` and pass each parsed line to " *
                "`extract_batch_text` individually.",
            ),
        )
    end

    return Pair{String,String}[]
end

function _generate_content_response_text(response::JSON3.Object)
    texts = String[]
    for candidate in get(response, :candidates, ())
        content = get(candidate, :content, nothing)
        content === nothing && continue
        for part in get(content, :parts, ())
            _is_answer_text(part) && push!(texts, String(part.text))
        end
    end
    return join(texts)
end

_generate_content_response_text(other) = ""
