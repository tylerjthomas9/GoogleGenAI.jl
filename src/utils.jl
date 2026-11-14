function string_key_dict(d::AbstractDict)
    return Dict{String,Any}(
        string(k) => (v isa AbstractDict ? string_key_dict(v) : v) for (k, v) in pairs(d)
    )
end

_compact(pairs::Pair...) = Dict{String,Any}(k => v for (k, v) in pairs if v !== nothing)

_key(d::AbstractDict, k::Symbol) = haskey(d, k) ? d[k] : get(d, String(k), nothing)

_model_path(name::AbstractString) = occursin('/', name) ? String(name) : "models/$name"

const MIME_TYPES = Dict(
    ".jpg" => "image/jpeg",
    ".jpeg" => "image/jpeg",
    ".png" => "image/png",
    ".gif" => "image/gif",
    ".webp" => "image/webp",
    ".heic" => "image/heic",
    ".heif" => "image/heif",
    ".bmp" => "image/bmp",
    ".tif" => "image/tiff",
    ".tiff" => "image/tiff",
    ".pdf" => "application/pdf",
    ".txt" => "text/plain",
    ".html" => "text/html",
    ".csv" => "text/csv",
    ".json" => "application/json",
    ".jsonl" => "application/jsonl",
    ".m4a" => "audio/m4a",
    ".wav" => "audio/wav",
    ".mp3" => "audio/mp3",
    ".aac" => "audio/aac",
    ".ogg" => "audio/ogg",
    ".flac" => "audio/flac",
    ".aiff" => "audio/aiff",
    ".mp4" => "video/mp4",
    ".mpeg" => "video/mpeg",
    ".mov" => "video/mov",
    ".avi" => "video/avi",
    ".webm" => "video/webm",
)

function _get_mime_type(file_path::AbstractString)
    return get(MIME_TYPES, lowercase(splitext(file_path)[2]), "application/octet-stream")
end

# Text parts that make up the model's answer (thought summaries excluded).
function _is_answer_text(part)
    return get(part, :text, nothing) !== nothing && get(part, :thought, false) !== true
end

_is_content(x) = x isa AbstractDict && _key(x, :parts) !== nothing
function _is_file(x)
    return x isa AbstractDict && _key(x, :uri) !== nothing && _key(x, :mimeType) !== nothing
end

function _file_part(uri, mime)
    return Dict{Symbol,Any}(:fileData => Dict(:fileUri => uri, :mimeType => mime))
end
_inline_part(mime, data::AbstractVector{UInt8}) = _inline_part(mime, base64encode(data))
function _inline_part(mime, data::AbstractString)
    return Dict{Symbol,Any}(:inlineData => Dict(:mimeType => mime, :data => data))
end

_to_part(text::AbstractString) = Dict{Symbol,Any}(:text => text)
function _to_part(part::AbstractDict)
    return _is_file(part) ? _file_part(_key(part, :uri), _key(part, :mimeType)) : part
end
function _to_part(media::NamedTuple)
    if haskey(media, :path)
        return _inline_part(
            get(media, :mime_type, _get_mime_type(media.path)), read(media.path)
        )
    elseif haskey(media, :data)
        return _inline_part(get(media, :mime_type, "image/png"), media.data)
    elseif haskey(media, :uri)
        haskey(media, :mime_type) ||
            throw(ArgumentError("`mime_type` is required for `uri` media"))
        return _file_part(media.uri, media.mime_type)
    end
    return throw(ArgumentError("media needs a `path`, `data`, or `uri` field"))
end

"""
Normalize a string, part, media, file or vector of these into one `Content` dict.
Dicts that already have `parts` pass through unchanged.
"""
function _to_content(x, role::Union{Nothing,String}="user")
    _is_content(x) && return x
    parts = x isa AbstractVector ? Any[_to_part(p) for p in x] : Any[_to_part(x)]
    content = Dict{Symbol,Any}(:parts => parts)
    role === nothing || (content[:role] = role)
    return content
end

"""
Normalize input into a vector of `Content` dicts. A vector whose elements are all contents is
a conversation and passes through; anything else becomes a single user turn.
"""
function _to_contents(x)
    x isa AbstractVector && !isempty(x) && all(_is_content, x) && return x
    return [_to_content(x)]
end

_system_instruction(x) = x === nothing ? nothing : _to_content(x, nothing)

function _with_media(contents, image_path::AbstractString, images::AbstractVector)
    contents = _to_contents(contents)
    media = Any[_to_part(m) for m in images]
    isempty(image_path) || pushfirst!(media, _to_part((path=image_path,)))
    isempty(media) && return contents
    last_turn = contents[end]
    turn = Dict{Symbol,Any}(
        :role => something(_key(last_turn, :role), "user"),
        :parts => Any[_key(last_turn, :parts)..., media...],
    )
    return Any[contents[1:(end - 1)]..., turn]
end
