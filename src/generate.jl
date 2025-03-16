
function _build_request_body(
    conversation::Vector{Dict{Symbol,Any}}, config::GenerateContentConfig
)
    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    body = Dict(
        "contents" => contents, "generationConfig" => _build_generation_config(config)
    )

    # Add optional fields
    config.tools !== nothing && (body["tools"] = config.tools)
    config.safety_settings !== nothing && (body["safetySettings"] = config.safety_settings)
    config.cached_content !== nothing && (body["cachedContent"] = config.cached_content)

    return body
end

"""
    generate_content(provider::AbstractGoogleProvider, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, prompt::String; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    
    generate_content(provider::AbstractGoogleProvider, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String, config=GenerateContentConfig()) -> NamedTuple
    generate_content(api_key::String, model_name::String, conversation::Vector{Dict{Symbol,Any}}; image_path::String, config=GenerateContentConfig()) -> NamedTuple

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
    - `response_status`: An integer representing the HTTP response status code.
    - `finish_reason`: A string indicating the reason why the generation process was finished.
"""
function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    conversation::Vector{Dict{Symbol,Any}};
    image_path::String="",
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
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        GoogleProvider(; api_key), model_name, conversation; image_path, config
    )
end

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        provider,
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
end

function generate_content(
    api_key::String,
    model_name::String,
    prompt::String;
    image_path::String="",
    config=GenerateContentConfig(),
)
    return generate_content(
        GoogleProvider(; api_key),
        model_name,
        [Dict(:role => "user", :parts => [Dict("text" => prompt)])];
        image_path,
        config,
    )
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

function generate_content(
    provider::AbstractGoogleProvider,
    model_name::String,
    contents::AbstractVector;
    image_path::String="",
    config=GenerateContentConfig(),
)
    parts = _convert_contents(contents)
    conversation = [Dict(:role => "user", :parts => parts)]
    return generate_content(provider, model_name, conversation; image_path, config)
end

function generate_content(
    api_key::String,
    model_name::String,
    contents::AbstractVector;
    image_path::String="",
    config=GenerateContentConfig(),
)
    parts = _convert_contents(contents)
    conversation = [Dict(:role => "user", :parts => parts)]
    return generate_content(
        GoogleProvider(; api_key), model_name, conversation; image_path, config
    )
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

    @async begin
        try
            # Make streaming request
            response = _request(
                provider,
                endpoint,
                :POST,
                body;
                stream=true,
                query=query,
                config.http_options...,
            )

            if response.status >= 400
                error_response = _parse_response(response)
                error_msg = get(error_response, :text, "HTTP Error $(response.status)")
                throw(ErrorException(error_msg))
            end

            # Process the SSE stream
            buffer = IOBuffer()

            # Use a Set to track exact chunks we've seen
            seen_chunks = Set{String}()

            for chunk in response.body
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
                        # Parse the SSE data chunk
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

                        # CRITICAL: Check if we've seen this exact text before
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
                            safety_ratings=get(chunk_data, :safetyRatings, Dict{Any,Any}()),
                            text=current_text,
                            status=response.status,
                            finish_reason=finish_reason,
                            usage_metadata=get(
                                chunk_data, :usageMetadata, Dict{Symbol,Any}()
                            ),
                        )

                        # Send the unique chunk to the channel
                        put!(processed_channel, parsed)

                        # If the generation is finished, break the loop
                        if finish_reason !== nothing
                            break
                        end

                    catch e
                        @warn "Failed to parse chunk: $e"
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

    # Now create a client-facing channel that will combine the chunks appropriately
    result_channel = Channel{NamedTuple}(32)

    @async begin
        try
            full_text = ""

            for chunk in processed_channel
                # If there's an error, pass it through
                if haskey(chunk, :error)
                    put!(result_channel, chunk)
                    break
                end

                # Update our combined text
                full_text = string(full_text, chunk.text)

                # Create a new chunk that includes the full text so far
                combined_chunk = (
                    candidates=chunk.candidates,
                    safety_ratings=chunk.safety_ratings,
                    text=chunk.text,  # Just the new part
                    full_text=full_text,  # Full text so far
                    status=chunk.status,
                    finish_reason=chunk.finish_reason,
                    usage_metadata=chunk.usage_metadata,
                )

                put!(result_channel, combined_chunk)

                # If this is the final chunk, we're done
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

# Helper function for client usage
function process_gemini_stream(stream)
    full_response = ""
    for chunk in stream
        if haskey(chunk, :error)
            error("Stream error: $(chunk.error)")
        end

        # Print just the new content
        print(chunk.text)
        flush(stdout)

        # Add to accumulated response
        full_response *= chunk.text

        # Check if we're done
        if chunk.finish_reason !== nothing
            break
        end
    end
    return full_response
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
