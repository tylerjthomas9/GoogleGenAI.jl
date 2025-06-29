function _format_system_instruction(instruction::String)
    return Dict("parts" => [Dict("text" => instruction)])
end

function _build_request_body(
    conversation::Vector{Dict{Symbol,Any}}, config::GenerateContentConfig
)
    body = Dict{String,Any}()
    if config.system_instruction !== nothing
        body["systemInstruction"] = _format_system_instruction(config.system_instruction)
    end
    body["contents"] = conversation
    body["generationConfig"] = _build_generation_config(config)
    body["tools"] = []

    # Add optional fields
    config.safety_settings !== nothing && (body["safetySettings"] = config.safety_settings)
    config.cached_content !== nothing && (body["cachedContent"] = config.cached_content)

    if config.tools !== nothing
        function_declarations = []

        for tool in config.tools
            if isa(tool, Function)
                try
                    decl = FunctionDeclaration(tool)
                    api_decl = to_api_function_declaration(decl)
                    push!(function_declarations, api_decl)
                catch e
                    @warn "Failed to convert function $(nameof(tool)) to declaration: $e"
                end
            elseif isa(tool, Dict)
                string_tool = Dict{String,Any}()
                for (k, v) in tool
                    string_tool[string(k)] = v
                end
                push!(body["tools"], string_tool)
            else
                @warn "Ignoring unsupported tool type: $(typeof(tool))"
            end
        end

        if !isempty(function_declarations)
            push!(
                body["tools"],
                Dict{String,Any}("functionDeclarations" => function_declarations),
            )
        end
    end

    if config.function_declarations !== nothing
        api_decls = [to_api_function_declaration(fd) for fd in config.function_declarations]
        found = false
        for tool in body["tools"]
            if haskey(tool, "functionDeclarations")
                append!(tool["functionDeclarations"], api_decls)
                found = true
                break
            end
        end

        if !found
            push!(body["tools"], Dict{String,Any}("functionDeclarations" => api_decls))
        end
    end

    if config.tool_config !== nothing
        tool_config_dict = Dict{String,Any}()

        if config.tool_config.function_calling_config !== nothing
            fc_config = config.tool_config.function_calling_config
            function_calling_dict = Dict{String,Any}("mode" => fc_config.mode)

            if fc_config.allowed_function_names !== nothing
                function_calling_dict["allowedFunctionNames"] =
                    fc_config.allowed_function_names
            end

            tool_config_dict["functionCallingConfig"] = function_calling_dict
        end

        body["toolConfig"] = tool_config_dict
    end

    return body
end

"""
    _parse_response(response) -> NamedTuple

Parse the API response into a structured format.

# Arguments
- `response`: The HTTP response object from the API.

# Returns
- NamedTuple with fields including candidates, safety_ratings, text, images, function_calls, etc.
"""
function _parse_response(response)
    body = JSON3.read(response.body)

    text_parts = String[]
    image_parts = []
    function_calls = []
    candidates = get(body, :candidates, [])

    finish_reason = nothing
    if !isempty(candidates)
        finish_reason = get(candidates[1], :finishReason, nothing)

        content = get(candidates[1], :content, nothing)
        if content !== nothing && haskey(content, :parts)
            for part in content.parts
                if haskey(part, :text) && part.text !== nothing
                    if !isempty(strip(part.text))
                        push!(text_parts, part.text)
                    end
                elseif haskey(part, :inlineData) && part.inlineData !== nothing
                    inline_data = part.inlineData
                    mime_type = get(inline_data, :mimeType, "image/png")
                    data = get(inline_data, :data, "")

                    data = strip(data)
                    if isempty(data)
                        continue
                    end
                    image_data = Base64.base64decode(data)
                    push!(image_parts, (data=image_data, mime_type=mime_type))
                elseif haskey(part, :functionCall) && part.functionCall !== nothing
                    fc = part.functionCall
                    name = get(fc, :name, "")
                    args_str = get(fc, :args, "{}")

                    # Parse args - it should be a JSON object already
                    args = if typeof(args_str) <: AbstractDict
                        Dict{String,Any}(String(k) => v for (k, v) in pairs(args_str))
                    else
                        JSON3.read(args_str, Dict{String,Any})
                    end

                    push!(function_calls, FunctionCall(name, args))
                end
            end
        end
    else
        text = get(body, :text, "")
        !isempty(text) && push!(text_parts, text)
    end

    full_text = join(text_parts, "")

    return (
        candidates=candidates,
        safety_ratings=get(body, :safetyRatings, Dict{Symbol,Any}()),
        text=full_text,
        images=image_parts,
        function_calls=isempty(function_calls) ? nothing : function_calls,
        response_status=response.status,
        finish_reason=finish_reason,
        usage_metadata=get(body, :usageMetadata, Dict{Symbol,Any}()),
    )
end

"""
    add_function_result_to_conversation(conversation::Vector{Dict{Symbol, Any}}, function_name::String, function_result::Any)

Adds a function result to a conversation for multi-turn function calling.

# Arguments
- `conversation::Vector{Dict{Symbol, Any}}`: The existing conversation.
- `function_name::String`: The name of the function that was called.
- `function_result::Any`: The result returned by the function.

# Returns
- Updated conversation vector with the function result added.
"""
function add_function_result_to_conversation(
    conversation::Vector{<:Dict}, function_name::String, function_result::Any
)
    result_str = if isa(function_result, String)
        function_result
    elseif isa(function_result, Dict) || isa(function_result, Vector)
        JSON3.write(function_result)
    else
        string(function_result)
    end

    function_response = Dict(
        :role => "function",
        :parts => [
            Dict(
                :functionResponse => Dict(
                    :name => function_name,
                    :response => Dict(:content => result_str),
                ),
            ),
        ],
    )

    push!(conversation, function_response)
    return conversation
end

function _convert_contents(contents::AbstractVector)
    parts = Vector{Dict{String,Union{String,Dict{String,String}}}}()
    for c in contents
        if isa(c, String)
            push!(parts, Dict("text" => c))
        elseif isa(c, JSON3.Object) || isa(c, Dict)
            cleaned_content = Dict{Symbol,String}(Symbol(k) => string(v) for (k, v) in c)
            push!(
                parts,
                Dict(
                    "file_data" => Dict(
                        "file_uri" => cleaned_content[:uri],
                        "mime_type" => cleaned_content[:mimeType],
                    ),
                ),
            )
        else
            error("Unsupported content type in contents vector: $(typeof(content))")
        end
    end
    return parts
end

"""
    generate_content(provider::AbstractGoogleProvider, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    
    generate_content(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; config=GenerateContentConfig()) -> NamedTuple

Generate content based on a combination of text prompt and an image (optional).

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt to accompany the image.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `NamedTuple`: A named tuple containing the following keys:
    - `candidates`: A vector of dictionaries, each representing a generation candidate.
    - `safety_ratings`: A dictionary containing safety ratings for the prompt feedback.
    - `text`: A string representing the concatenated text from all candidates.
    - `function_calls`: Optional vector of function calls from the model.
    - `response_status`: An integer representing the HTTP response status code.
    - `finish_reason`: A string indicating the reason why the generation process was finished.
"""
function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    config::GenerateContentConfig=GenerateContentConfig(),
)
    endpoint = "models/$model_name:generateContent"
    body = _build_request_body(conversation, config)
    response = _request(provider, endpoint, :POST, body; config.http_options...)
    return _parse_response(response)
end

function generate_content(
    api_key::String,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    config=GenerateContentConfig(),
)
    return generate_content(GoogleProvider(; api_key), model_name, conversation; config)
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    if isempty(image_path)
        conversation = [Dict(:role => "user", :parts => [Dict("text" => prompt)])]
    else
        ext = lowercase(splitext(image_path)[2])
        if ext in [".jpg", ".jpeg"]
            mime_type = "image/jpeg"
        elseif ext == ".png"
            mime_type = "image/png"
        else
            throw("Unkown image file format $image_path")
        end
        image_data = open(base64encode, image_path)
        conversation = [
            Dict(
                :role => "user",
                :parts => [
                    Dict("text" => prompt),
                    Dict(
                        "inline_data" =>
                            Dict("mime_type" => mime_type, "data" => image_data),
                    ),
                ],
            ),
        ]
    end
    return generate_content(provider, model_name, conversation; config)
end

function generate_content(
    api_key::String,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, prompt; image_path, config
    )
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents::AbstractVector;
    image_path::String="",
    config=GenerateContentConfig(),
)
    parts = _convert_contents(contents)
    if isempty(image_path)
        conversation = [Dict(:role => "user", :parts => parts)]
    else
        ext = lowercase(splitext(image_path)[2])
        if ext in [".jpg", ".jpeg"]
            mime_type = "image/jpeg"
        elseif ext == ".png"
            mime_type = "image/png"
        else
            throw("Unkown image file format $image_path")
        end
        image_data = open(base64encode, image_path)
        conversation = [
            Dict(
                :role => "user",
                :parts => [
                    parts...,
                    Dict(
                        "inline_data" =>
                            Dict("mime_type" => mime_type, "data" => image_data),
                    ),
                ],
            ),
        ]
    end
    return generate_content(provider, model_name, conversation; config)
end

function generate_content(
    api_key::String,
    model_name::String,
    contents::AbstractVector;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(GoogleProvider(; api_key), model_name, contents; config)
end

"""
    generate_content_stream(provider::AbstractGoogleProvider, model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> Channel
    generate_content_stream(api_key::String, model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> Channel
    
    generate_content_stream(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String="", config=GenerateContentConfig()) -> Channel
    generate_content_stream(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String="", config=GenerateContentConfig()) -> Channel

Generate content in a streaming fashion, returning partial results as they become available.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance for API requests.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt to accompany the image.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `Channel`: A channel that yields partial text responses as they become available.
  Each item in the channel is a named tuple with the following fields:
  - `text::String`: The partial text response.
  - `finish_reason::Union{String,Nothing}`: The reason why generation stopped, if applicable.
  - `is_final::Bool`: Whether this is the final chunk of the response.
"""
function generate_content_stream(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config::GenerateContentConfig=GenerateContentConfig(),
)
    endpoint = "models/$model_name:streamGenerateContent"
    body = _build_request_body(conversation, config)

    # Create a channel for processed chunks
    processed_channel = Channel{NamedTuple}(32)

    query = Dict(
        "key" => provider.api_key,
        "alt" => "sse",  # Critical parameter for SSE format
    )

    url = "$(provider.base_url)/$(provider.api_version)/$endpoint"
    headers = Dict("Content-Type" => "application/json")

    @async begin
        try
            # Use HTTP.open for true streaming
            HTTP.open(
                "POST", url; headers=headers, query=query, config.http_options...
            ) do stream
                # Write the request body
                write(stream, JSON3.write(body))
                HTTP.closewrite(stream)

                # Start reading the response
                response = HTTP.startread(stream)

                if response.status >= 400
                    error_msg = "HTTP Error $(response.status): $(String(response.body))"
                    throw(ErrorException(error_msg))
                end

                # Process the SSE stream
                buffer = IOBuffer()
                seen_chunks = Set{String}()

                while !eof(stream)
                    chunk = readavailable(stream)
                    if isempty(chunk)
                        sleep(0.005)  # Small delay to prevent CPU spinning
                        continue
                    end

                    write(buffer, chunk)
                    seekstart(buffer)

                    while !eof(buffer)
                        line = readline(buffer; keep=true)

                        # Save partial line for next chunk
                        if !endswith(line, '\n')
                            partial = take!(buffer)
                            write(buffer, partial)
                            break
                        end

                        line = strip(line)
                        if !startswith(line, "data: ")
                            continue
                        end

                        # Extract data
                        data_str = SubString(line, 7)  # Remove "data: " prefix
                        if data_str == "[DONE]"
                            break
                        end

                        try
                            chunk_data = JSON3.read(data_str)

                            # Extract the text from the JSON structure
                            if !haskey(chunk_data, :candidates) ||
                                isempty(chunk_data.candidates)
                                continue
                            end

                            content = get(chunk_data.candidates[1], :content, nothing)
                            if content === nothing ||
                                !haskey(content, :parts) ||
                                isempty(content.parts)
                                continue
                            end

                            # Get the text from the first part
                            current_text = get(content.parts[1], :text, "")
                            if current_text in seen_chunks
                                continue  # Skip duplicate chunks entirely
                            end

                            # Add this chunk to our seen set
                            push!(seen_chunks, current_text)

                            # Create a parsed response
                            finish_reason = get(
                                chunk_data.candidates[1], :finishReason, nothing
                            )

                            parsed = (
                                candidates=chunk_data.candidates,
                                safety_ratings=get(
                                    chunk_data, :safetyRatings, Dict{Any,Any}()
                                ),
                                text=current_text,
                                status=response.status,
                                finish_reason=finish_reason,
                                usage_metadata=get(
                                    chunk_data, :usageMetadata, Dict{Symbol,Any}()
                                ),
                            )

                            # Send the unique chunk to the channel
                            put!(processed_channel, parsed)

                        catch e
                            @warn "Failed to parse chunk: $e"
                        end
                    end
                end
            end

        catch e
            put!(
                processed_channel, (error=e, text="", finish_reason="ERROR", is_final=true)
            )
        finally
            close(processed_channel)
        end
    end

    result_channel = Channel{NamedTuple}(32)

    @async begin
        try
            full_text = ""
            function_calls = []

            for chunk in processed_channel
                if haskey(chunk, :error)
                    put!(result_channel, chunk)
                    break
                end
                full_text = string(full_text, chunk.text)

                # Extract function calls if present in the chunk
                if haskey(chunk, :candidates) && !isempty(chunk.candidates)
                    content = get(chunk.candidates[1], :content, nothing)
                    if content !== nothing && haskey(content, :parts)
                        for part in content.parts
                            if haskey(part, :functionCall) && part.functionCall !== nothing
                                fc = part.functionCall
                                name = get(fc, :name, "")
                                args_str = get(fc, :args, "{}")

                                # Parse args
                                args = if typeof(args_str) <: AbstractDict
                                    Dict{String,Any}(
                                        String(k) => v for (k, v) in pairs(args_str)
                                    )
                                else
                                    JSON3.read(args_str, Dict{String,Any})
                                end

                                push!(function_calls, FunctionCall(name, args))
                            end
                        end
                    end
                end

                combined_chunk = (
                    candidates=chunk.candidates,
                    safety_ratings=chunk.safety_ratings,
                    text=chunk.text,  # Just the new part
                    full_text=full_text,  # Full text so far
                    function_calls=isempty(function_calls) ? nothing : function_calls,
                    status=chunk.status,
                    finish_reason=chunk.finish_reason,
                    usage_metadata=chunk.usage_metadata,
                )

                put!(result_channel, combined_chunk)

                if chunk.finish_reason !== nothing
                    break
                end
            end
        finally
            close(result_channel)
        end
    end

    return result_channel
end

function generate_content_stream(
    api_key::String,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        GoogleProvider(; api_key), model_name, conversation; image_path, config
    )
end

function generate_content_stream(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        provider,
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

function generate_content_stream(
    api_key::String,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        GoogleProvider(; api_key),
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

"""
    count_tokens(provider::AbstractGoogleProvider, model_name::String, prompt::String) -> Int
    count_tokens(api_key::String, model_name::String, prompt::String) -> Int

Calculate the number of tokens generated by the specified model for a given prompt string.

# Arguments
- `provider::AbstractGoogleProvider`: The provider instance containing API key and base URL information.
- `api_key::String`: Your Google API key as a string. 
- `model_name::String`: The name of the model to use for generating content. 
- `prompt::String`: The prompt prompt based on which the text is generated.

# Returns
- `Int`: The total number of tokens that the given prompt string would be broken into by the specified model's tokenizer.
"""
function count_tokens(provider::AbstractGoogleProvider, model_name::String, prompt::String)
    endpoint = "models/$model_name:countTokens"
    body = Dict("contents" => [Dict("parts" => [Dict("text" => prompt)])])
    response = _request(provider, endpoint, :POST, body)
    total_tokens = get(JSON3.read(response.body), "totalTokens", 0)
    return total_tokens
end

function count_tokens(api_key::String, model_name::String, prompt::String)
    return count_tokens(GoogleProvider(; api_key), model_name, prompt)
end

"""
    generate_content(model_name::String, conversation::Vector{Dict{Symbol,Any}}; config=GenerateContentConfig()) -> NamedTuple

Generate content using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The model to use for content generation.
- `conversation::Vector{Dict{Symbol,Any}}`: The conversation history.

# Keyword Arguments
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `NamedTuple`: Same as other generate_content functions.
"""
function generate_content(
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    config=GenerateContentConfig(),
)
    return generate_content(GoogleProvider(), model_name, conversation; config)
end

"""
    generate_content(model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> NamedTuple

Generate content using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `NamedTuple`: Same as other generate_content functions.
"""
function generate_content(
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(GoogleProvider(), model_name, prompt; image_path, config)
end

"""
    generate_content(model_name::String, contents::AbstractVector; image_path::String="", config=GenerateContentConfig()) -> NamedTuple

Generate content using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The model to use for content generation.
- `contents::AbstractVector`: The contents vector.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `NamedTuple`: Same as other generate_content functions.
"""
function generate_content(
    model_name::String,
    contents::AbstractVector;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(GoogleProvider(), model_name, contents; image_path, config)
end

"""
    generate_content_stream(model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String="", config=GenerateContentConfig()) -> Channel

Generate streaming content using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The model to use for content generation.
- `conversation::Vector{Dict{Symbol,Any}}`: The conversation history.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `Channel`: Same as other generate_content_stream functions.
"""
function generate_content_stream(
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(
        GoogleProvider(), model_name, conversation; image_path, config
    )
end

"""
    generate_content_stream(model_name::String, prompt::String; image_path::String="", config=GenerateContentConfig()) -> Channel

Generate streaming content using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The model to use for content generation.
- `prompt::String`: The text prompt.

# Keyword Arguments
- `image_path::String` (optional): The path to the image file to include in the request.
- `config::GenerateContentConfig` (optional): Configuration for the generation request.

# Returns
- `Channel`: Same as other generate_content_stream functions.
"""
function generate_content_stream(
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content_stream(GoogleProvider(), model_name, prompt; image_path, config)
end

"""
    count_tokens(model_name::String, prompt::String) -> Int

Count tokens using automatic API key detection from environment variables.

# Arguments
- `model_name::String`: The name of the model to use for token counting.  
- `prompt::String`: The prompt to count tokens for.

# Returns
- `Int`: The total number of tokens.
"""
function count_tokens(model_name::String, prompt::String)
    return count_tokens(GoogleProvider(), model_name, prompt)
end
