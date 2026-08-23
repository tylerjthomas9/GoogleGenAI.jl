module GoogleGenAI

using Base64
using JSON3
using HTTP

include("utils.jl")
include("core.jl")
include("functions.jl")
include("generate.jl")
include("chat.jl")
include("cache.jl")
include("file.jl")
include("embed.jl")
include("batch.jl")

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
    Chat,
    create_chat,
    send_message,
    send_message_stream,
    get_history,
    clear_history!,
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
    is_native_tool,
    BatchRequest,
    create_batch_job,
    get_batch_job,
    poll_batch_job,
    cancel_batch_job,
    delete_batch_job,
    download_batch_output_file,
    extract_batch_text

end # module GoogleGenAI
