abstract type AbstractQuestion end
abstract type AbstractAnswer end

"""
    NoulQuestion(instructions; true_description=nothing, false_description=nothing)

A binary (true / false) decision question.
"""
struct NoulQuestion <: AbstractQuestion
    instructions::String
    true_description::Union{Nothing,String}
    false_description::Union{Nothing,String}
end
function NoulQuestion(
    instructions::AbstractString; true_description=nothing, false_description=nothing
)
    return NoulQuestion(instructions, true_description, false_description)
end

"""
    ChoiceQuestion(instructions, options::AbstractVector{<:AbstractString})
    ChoiceQuestion(instructions, criteria::AbstractDict)

A single-select categorical decision question over mutually exclusive options. `criteria`
maps each option ID to an optional description (2 to 255 options).
"""
struct ChoiceQuestion <: AbstractQuestion
    instructions::String
    criteria::Dict{String,String}
    function ChoiceQuestion(instructions::AbstractString, criteria::AbstractDict)
        length(criteria) >= 2 ||
            throw(ArgumentError("ChoiceQuestion needs at least 2 options"))
        return new(instructions, Dict(String(k) => String(v) for (k, v) in criteria))
    end
end
function ChoiceQuestion(
    instructions::AbstractString, options::AbstractVector{<:AbstractString}
)
    return ChoiceQuestion(instructions, Dict(o => "" for o in options))
end

"""
    ScoreQuestion(instructions, criteria::AbstractVector{<:AbstractString})

A quantitative rubric scoring question over discrete levels, ordered lowest to highest
(2 to 255 levels).
"""
struct ScoreQuestion <: AbstractQuestion
    instructions::String
    criteria::Vector{String}
    function ScoreQuestion(
        instructions::AbstractString, criteria::AbstractVector{<:AbstractString}
    )
        length(criteria) >= 2 ||
            throw(ArgumentError("ScoreQuestion needs at least 2 levels"))
        return new(instructions, criteria)
    end
end

"""
    NoulAnswer

Calibrated result for a [`NoulQuestion`](@ref). `noul` is `P(true)` in `[0.01, 0.99]`;
`confidence` is `max(noul, 1 - noul)`.
"""
struct NoulAnswer <: AbstractAnswer
    noul::Float64
    confidence::Float64
end

"""
    ChoiceAnswer

Calibrated result for a [`ChoiceQuestion`](@ref). `choice` is the option with the highest
probability, `probabilities` maps each option to its probability, and `confidence` is the
probability of `choice`.
"""
struct ChoiceAnswer <: AbstractAnswer
    choice::String
    probabilities::Dict{String,Float64}
    confidence::Float64
end

"""
    ScoreAnswer

Calibrated result for a [`ScoreQuestion`](@ref). `score` is the probability-weighted
expected **0-based** level index, `legend` and `probabilities` list each level and its
probability in order, and `confidence` is the highest probability.
"""
struct ScoreAnswer <: AbstractAnswer
    score::Float64
    legend::Vector{String}
    probabilities::Vector{Float64}
    confidence::Float64
end

function _question_dict(q::NoulQuestion)
    d = Dict{String,Any}("type" => "noul", "instructions" => q.instructions)
    criteria = Dict{String,String}()
    isnothing(q.true_description) || (criteria["true_description"] = q.true_description)
    isnothing(q.false_description) || (criteria["false_description"] = q.false_description)
    isempty(criteria) || (d["criteria"] = criteria)
    return d
end
function _question_dict(q::ChoiceQuestion)
    return Dict{String,Any}(
        "type" => "choice", "instructions" => q.instructions, "criteria" => q.criteria
    )
end
function _question_dict(q::ScoreQuestion)
    return Dict{String,Any}(
        "type" => "score", "instructions" => q.instructions, "criteria" => q.criteria
    )
end
_question_dict(q::AbstractDict) = q

function _questions_dict(questions::Union{NamedTuple,AbstractDict})
    isempty(questions) && throw(ArgumentError("questions cannot be empty"))
    return Dict(string(k) => _question_dict(v) for (k, v) in pairs(questions))
end

_decision_input(input::AbstractString) = input
_decision_input(input::AbstractVector) = [_input_block(x) for x in input]

_input_block(text::AbstractString) = Dict("type" => "text", "text" => text)
_input_block(block::AbstractDict) = block
function _input_block(file::JSON3.Object)
    haskey(file, :uri) && haskey(file, :mimeType) || return file
    return _media_block(String(file.mimeType); uri=String(file.uri))
end
function _input_block(media::NamedTuple)
    resolution = get(media, :resolution, nothing)
    if haskey(media, :path)
        mime = get(media, :mime_type, _get_mime_type(media.path))
        return _media_block(mime; data=open(base64encode, media.path), resolution)
    end
    haskey(media, :mime_type) ||
        throw(ArgumentError("mime_type is required for `data` and `uri` inputs"))
    if haskey(media, :data)
        data = media.data isa AbstractVector{UInt8} ? base64encode(media.data) : media.data
        return _media_block(media.mime_type; data, resolution)
    elseif haskey(media, :uri)
        return _media_block(media.mime_type; uri=media.uri, resolution)
    end
    return throw(ArgumentError("media input needs a `path`, `data`, or `uri` field"))
end

function _media_block(mime::AbstractString; data=nothing, uri=nothing, resolution=nothing)
    block = Dict{String,Any}("type" => _media_type(mime), "mime_type" => mime)
    isnothing(data) || (block["data"] = data)
    isnothing(uri) || (block["uri"] = uri)
    isnothing(resolution) || (block["resolution"] = resolution)
    return block
end

function _media_type(mime::AbstractString)
    mime == "application/pdf" && return "document"
    for kind in ("image", "audio", "video")
        startswith(mime, "$kind/") && return kind
    end
    return throw(ArgumentError("unsupported MIME type for decisions input: $mime"))
end

function _build_decision_body(model_name::String, input, questions)
    return Dict(
        "model" => model_name,
        "input" => _decision_input(input),
        "questions" => _questions_dict(questions),
    )
end

function _parse_answer(answer)
    type = get(answer, :type, nothing)
    if type == "noul"
        return NoulAnswer(answer.noul, answer.confidence)
    elseif type == "choice"
        probabilities = Dict(String(k) => Float64(v) for (k, v) in answer.probabilities)
        return ChoiceAnswer(answer.choice, probabilities, answer.confidence)
    elseif type == "score"
        levels = Symbol.(0:(length(answer.legend) - 1))
        return ScoreAnswer(
            answer.score,
            [String(answer.legend[l]) for l in levels],
            [Float64(get(answer.probabilities, l, 0.0)) for l in levels],
            answer.confidence,
        )
    end
    return answer
end

function _parse_decision_response(response)
    body = JSON3.read(response.body)
    answers = Dict{String,Union{AbstractAnswer,JSON3.Object}}(
        String(k) => _parse_answer(v) for (k, v) in get(body, :answers, Dict())
    )
    return (
        answers=answers,
        usage=get(body, :usage, nothing),
        prompt_feedback=get(body, :prompt_feedback, nothing),
        response_status=response.status,
    )
end

"""
    create_decision([provider | api_key], model_name, input, questions; http_kwargs=NamedTuple()) -> NamedTuple

Evaluates a batch of structured decision questions against a shared `input`.

`input` is a string or a vector of strings and media. Media is a NamedTuple with a `path`,
`data`, or `uri` field plus optional `mime_type` and `resolution`, or a file returned by
[`upload_file`](@ref). `questions` maps question IDs to [`NoulQuestion`](@ref),
[`ChoiceQuestion`](@ref), or [`ScoreQuestion`](@ref) values.

Returns `(answers, usage, prompt_feedback, response_status)`, where `answers` maps each
question ID to a [`NoulAnswer`](@ref), [`ChoiceAnswer`](@ref), or [`ScoreAnswer`](@ref).

```julia
r = create_decision("gemini-decision-1", "The mug arrived cracked.",
    (damaged=NoulQuestion("Did the item arrive damaged?"),))
r.answers["damaged"].noul
```
"""
function create_decision(
    provider::AbstractGoogleProvider,
    model_name::String,
    input,
    questions;
    http_kwargs=NamedTuple(),
)
    body = _build_decision_body(model_name, input, questions)
    response = _request(provider, "decisions", :POST, body; http_kwargs...)
    return _parse_decision_response(response)
end

function create_decision(
    api_key::String, model_name::String, input, questions; http_kwargs=NamedTuple()
)
    return create_decision(
        GoogleProvider(; api_key), model_name, input, questions; http_kwargs
    )
end

function create_decision(model_name::String, input, questions; http_kwargs=NamedTuple())
    return create_decision(GoogleProvider(), model_name, input, questions; http_kwargs)
end
