function _get_mime_type(file_path::String)::String
    ext = lowercase(splitext(file_path)[2])
    mime_map = Dict(
        # Image types
        ".jpg" => "image/jpeg",
        ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",

        # Document types
        ".pdf" => "application/pdf",

        # Text types
        ".txt" => "text/plain",
        ".html" => "text/html",
        ".csv" => "text/csv",

        # Audio types
        ".m4a" => "audio/m4a",
        ".wav" => "audio/wav",

        # Video types
        ".mp4" => "video/mp4",
    )
    return get(mime_map, ext, "application/octet-stream")
end
