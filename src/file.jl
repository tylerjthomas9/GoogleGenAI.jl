
function _get_mime_type(file_path::String)::String
    ext = lowercase(splitext(file_path)[2])
    if ext in [".jpg", ".jpeg"]
        return "image/jpeg"
    elseif ext == ".png"
        return "image/png"
    elseif ext == ".gif"
        return "image/gif"
    elseif ext == ".pdf"
        return "application/pdf"
    elseif ext == ".txt"
        return "text/plain"
    elseif ext == ".html"
        return "text/html"
    elseif ext == ".csv"
        return "text/csv"
    elseif ext == ".m4a"
        return "audio/m4a"
    elseif ext == ".wav"
        return "audio/wav"
    elseif ext == ".mp4"
        return "video/mp4"
    else
        throw("Unknown mime type for file $file_path ($ext). Please specify `mime_type`.")
    end
end

"""
    upload_file(provider::AbstractGoogleProvider, file_path::String; display_name::String="", mime_type::String="application/octet-stream", http_kwargs=NamedTuple()) -> JSON3.Object

Uploads a file using the resumable upload protocol.
"""
function upload_file(
    provider::AbstractGoogleProvider,
    file_path::String;
    display_name::String="",
    mime_type::String="",
    http_kwargs=NamedTuple(),
)
    if mime_type == ""
        mime_type = _get_mime_type(file_path)
    end
    if display_name == ""
        display_name = basename(file_path)
    end

    # Step 1: Initiate the upload
    url = "$(provider.base_url)/upload/$(provider.api_version)/files?key=$(provider.api_key)"
    file_size = filesize(file_path)
    headers = Dict(
        "X-Goog-Upload-Protocol" => "resumable",
        "X-Goog-Upload-Command" => "start",
        "X-Goog-Upload-Header-Content-Length" => string(file_size),
        "X-Goog-Upload-Header-Content-Type" => mime_type,
        "Content-Type" => "application/json",
    )
    body = Dict("file" => Dict("displayName" => display_name))
    serialized_body = JSON3.write(body)

    response = HTTP.request("POST", url, headers, serialized_body; http_kwargs...)
    if response.status >= 400
        status_error(response, String(response.body))
    end

    # Extract the upload URL
    upload_url = get(Dict(response.headers), "X-Goog-Upload-URL", nothing)
    if upload_url === nothing
        error("Failed to obtain upload URL")
    end

    # Step 2: Upload the file content
    file_content = read(file_path)
    upload_headers = Dict(
        "Content-Length" => string(file_size),
        "X-Goog-Upload-Offset" => "0",
        "X-Goog-Upload-Command" => "upload, finalize",
    )

    response = HTTP.request("PUT", upload_url, upload_headers, file_content; http_kwargs...)
    if response.status >= 400
        status_error(response, String(response.body))
    end

    return JSON3.read(String(response.body))[:file]
end

# Overload for direct API key usage (unchanged)
function upload_file(
    api_key::String,
    file_path::String;
    display_name::String="",
    mime_type::String="",
    http_kwargs=NamedTuple(),
)
    return upload_file(
        GoogleProvider(; api_key),
        file_path;
        display_name=display_name,
        mime_type=mime_type,
        http_kwargs=http_kwargs,
    )
end

"""
    get_file(provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()) -> Any

Retrieves metadata for the file specified by its resource name (e.g. "files/abc-123").
"""
function get_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    response = _request(provider, file_name, :GET, Dict(); http_kwargs...)
    return JSON3.read(String(response.body))
end

function get_file(api_key::String, file_name::String; http_kwargs=NamedTuple())
    return get_file(GoogleProvider(; api_key), file_name; http_kwargs=http_kwargs)
end

"""
    list_files(provider::AbstractGoogleProvider; page_size::Int=10, page_token::String="", http_kwargs=NamedTuple()) -> JSON3.Array

Lists file metadata for files owned by your project. Use `page_size` and `page_token` for pagination.
"""
function list_files(
    provider::AbstractGoogleProvider;
    page_size::Int=10,
    page_token::String="",
    http_kwargs=NamedTuple(),
)
    # Build the URL with query parameters.
    url = "$(provider.base_url)/$(provider.api_version)/files?key=$(provider.api_key)&pageSize=$(page_size)"
    if page_token != ""
        url *= "&pageToken=$(page_token)"
    end
    headers = Dict("Content-Type" => "application/json")

    response = HTTP.request(:GET, url, headers, ""; http_kwargs...)
    if response.status >= 400
        status_error(response, String(response.body))
    end
    return JSON3.read(String(response.body))[:files]
end

function list_files(
    api_key::String; page_size::Int=10, page_token::String="", http_kwargs=NamedTuple()
)
    return list_files(
        GoogleProvider(; api_key);
        page_size=page_size,
        page_token=page_token,
        http_kwargs=http_kwargs,
    )
end

"""
    delete_file(provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()) -> Int

Deletes the file specified by its resource name (e.g. "files/abc-123") and returns the HTTP status code.
"""
function delete_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    response = _request(provider, file_name, :DELETE, Dict(); http_kwargs...)
    return response.status
end

function delete_file(api_key::String, file_name::String; http_kwargs=NamedTuple())
    return delete_file(GoogleProvider(; api_key), file_name; http_kwargs=http_kwargs)
end
