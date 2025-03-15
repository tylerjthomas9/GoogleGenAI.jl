module GoogleGenAI

using Base64
using JSON3
using HTTP

include("core.jl")
include("generate.jl")
include("cache.jl")
include("file.jl")
include("embed.jl")

export GoogleProvider,
    SafetySetting,
    GenerateContentConfig,
    generate_content,
    generate_content_stream,
    count_tokens,
    embed_content,
    list_models,
    create_cached_content,
    list_cached_content,
    get_cached_content,
    update_cached_content,
    delete_cached_content,
    upload_file,
    get_file,
    list_files,
    delete_file

end # module GoogleGenAI
