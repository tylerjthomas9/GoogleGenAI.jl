"""
    upload_file([provider | api_key], file_path; display_name="", mime_type="", http_kwargs=NamedTuple()) -> JSON3.Object

Uploads a file using the resumable upload protocol and returns its metadata. The MIME type
is inferred from the extension when not given.
"""
function upload_file(
    provider::AbstractGoogleProvider,
    file_path::String;
    display_name::String="",
    mime_type::String="",
    http_kwargs=NamedTuple(),
)
    isempty(mime_type) && (mime_type = _get_mime_type(file_path))
    isempty(display_name) && (display_name = basename(file_path))
    file_size = string(filesize(file_path))

    url = "$(provider.base_url)/upload/$(provider.api_version)/files"
    headers = merge(
        _headers(provider),
        Dict(
            "X-Goog-Upload-Protocol" => "resumable",
            "X-Goog-Upload-Command" => "start",
            "X-Goog-Upload-Header-Content-Length" => file_size,
            "X-Goog-Upload-Header-Content-Type" => mime_type,
        ),
    )
    body = JSON3.write(Dict("file" => Dict("displayName" => display_name)))
    response = _send(provider, "POST", url; headers, body, http_kwargs...)

    upload_url = HTTP.header(response, "X-Goog-Upload-URL", "")
    isempty(upload_url) && error("Failed to obtain upload URL")

    upload_headers = Dict(
        "Content-Length" => file_size,
        "X-Goog-Upload-Offset" => "0",
        "X-Goog-Upload-Command" => "upload, finalize",
    )
    response = _send(
        provider,
        "PUT",
        upload_url;
        headers=upload_headers,
        body=read(file_path),
        http_kwargs...,
    )
    return JSON3.read(response.body)[:file]
end

function upload_file(api_key::String, file_path::String; kwargs...)
    return upload_file(GoogleProvider(; api_key), file_path; kwargs...)
end
function upload_file(file_path::String; kwargs...)
    return upload_file(GoogleProvider(), file_path; kwargs...)
end

"""
    get_file([provider | api_key], file_name; http_kwargs=NamedTuple()) -> JSON3.Object

Retrieves metadata for the file specified by its resource name (e.g. "files/abc-123").
"""
function get_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    return _request_json(provider, file_name, :GET; http_kwargs...)
end

function get_file(api_key::String, file_name::String; kwargs...)
    return get_file(GoogleProvider(; api_key), file_name; kwargs...)
end
get_file(file_name::String; kwargs...) = get_file(GoogleProvider(), file_name; kwargs...)

"""
    list_files([provider | api_key]; page_size=100, http_kwargs=NamedTuple()) -> Vector

Lists metadata for every file owned by your project, fetching all pages. `page_size` is the
number of files requested per page (at most 100).
"""
function list_files(
    provider::AbstractGoogleProvider; page_size::Int=100, http_kwargs=NamedTuple()
)
    return _paginate(provider, "files", :files; page_size, http_kwargs...)
end

list_files(api_key::String; kwargs...) = list_files(GoogleProvider(; api_key); kwargs...)
list_files(; kwargs...) = list_files(GoogleProvider(); kwargs...)

"""
    delete_file([provider | api_key], file_name; http_kwargs=NamedTuple()) -> Int

Deletes the file specified by its resource name (e.g. "files/abc-123") and returns the HTTP status code.
"""
function delete_file(
    provider::AbstractGoogleProvider, file_name::String; http_kwargs=NamedTuple()
)
    return _request(provider, file_name, :DELETE; http_kwargs...).status
end

function delete_file(api_key::String, file_name::String; kwargs...)
    return delete_file(GoogleProvider(; api_key), file_name; kwargs...)
end
function delete_file(file_name::String; kwargs...)
    return delete_file(GoogleProvider(), file_name; kwargs...)
end

"""
    download_file([provider | api_key], file_name; path=nothing, http_kwargs=NamedTuple())

Downloads the contents of a file the API generated, such as a batch job's output file.
Returns the bytes, or writes them to `path` and returns `path`. Files you uploaded cannot be
downloaded.
"""
function download_file(
    provider::AbstractGoogleProvider,
    file_name::String;
    path::Union{Nothing,AbstractString}=nothing,
    http_kwargs=NamedTuple(),
)
    url = "$(provider.base_url)/download/$(provider.api_version)/$file_name:download"
    response = _send(provider, :GET, url; query=Dict("alt" => "media"), http_kwargs...)
    path === nothing && return response.body
    write(path, response.body)
    return path
end

function download_file(api_key::String, file_name::String; kwargs...)
    return download_file(GoogleProvider(; api_key), file_name; kwargs...)
end
function download_file(file_name::String; kwargs...)
    return download_file(GoogleProvider(), file_name; kwargs...)
end
