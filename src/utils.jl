function string_key_dict(d::AbstractDict)
    return Dict{String,Any}(
        string(k) => (v isa AbstractDict ? string_key_dict(v) : v) for (k, v) in pairs(d)
    )
end

function _get_mime_type(file_path::String)::String
    ext = lowercase(splitext(file_path)[2])
    mime_map = Dict(
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
    return get(mime_map, ext, "application/octet-stream")
end
