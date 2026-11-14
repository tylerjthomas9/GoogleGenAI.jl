module GoogleGenAI

using Base64
using Dates
using JSON3
using HTTP

include("utils.jl")
include("core.jl")
include("request.jl")
include("models.jl")
include("functions.jl")
include("generate.jl")
include("chat.jl")
include("cache.jl")
include("file.jl")
include("embed.jl")
include("decisions.jl")
include("batch.jl")
include("live.jl")

export GoogleProvider,
    RetryOptions,
    GeminiAPIError,
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
    NoulQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    create_decision,
    list_models,
    get_model,
    create_cached_content,
    list_cached_content,
    get_cached_content,
    update_cached_content,
    delete_cached_content,
    upload_file,
    get_file,
    list_files,
    delete_file,
    download_file,
    ToolType,
    GOOGLE_SEARCH,
    CODE_EXECUTION,
    FUNCTION_CALLING,
    URL_CONTEXT,
    GOOGLE_MAPS,
    FILE_SEARCH,
    COMPUTER_USE,
    MCP_SERVERS,
    is_native_tool,
    google_search_tool,
    code_execution_tool,
    url_context_tool,
    google_maps_tool,
    file_search_tool,
    computer_use_tool,
    BatchRequest,
    create_batch_job,
    create_embeddings_batch_job,
    list_batch_jobs,
    get_batch_job,
    poll_batch_job,
    cancel_batch_job,
    delete_batch_job,
    download_batch_output_file,
    extract_batch_text,
    LiveSession,
    connect_live,
    send_realtime_input,
    send_client_content,
    send_tool_response,
    recv_event,
    receive_text,
    close_live!

end # module GoogleGenAI
