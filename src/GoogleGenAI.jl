module GoogleGenAI

using Base64
using JSON3
using HTTP

include("utils.jl")
include("core.jl")
include("functions.jl")
include("generate.jl")
include("cache.jl")
include("file.jl")
include("embed.jl")

export GoogleProvider,
    SafetySetting,
    ThinkingConfig,
    GenerateContentConfig,
    FunctionCall,
    FunctionParameter,
    FunctionDeclaration,
    FunctionCallingConfig,
    ToolConfig,
    add_function_result_to_conversation,
    execute_parallel_function_calls,
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
    delete_file,
    ToolType,
    GOOGLE_SEARCH,
    CODE_EXECUTION,
    FUNCTION_CALLING,
    is_native_tool

end # module GoogleGenAI
