
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

    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    generation_config = _build_generation_config(config)

    body = Dict("contents" => contents, "generationConfig" => generation_config)

    if config.tools !== nothing
        body["tools"] = config.tools
    end

    if config.safety_settings !== nothing
        body["safetySettings"] = config.safety_settings
    end

    if config.cached_content !== nothing
        body["cachedContent"] = config.cached_content
    end

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

    contents = []
    for turn in conversation
        role = turn[:role]
        parts = turn[:parts]
        push!(contents, Dict("role" => role, "parts" => parts))
    end

    generation_config = _build_generation_config(config)

    body = Dict("contents" => contents, "generationConfig" => generation_config)

    if config.safety_settings !== nothing
        body["safetySettings"] = config.safety_settings
    end

    if config.cached_content !== nothing
        body["cachedContent"] = config.cached_content
    end

    # Create a channel to stream the results
    result_channel = Channel{NamedTuple}(32)

    # Start a task to handle the streaming
    @async begin
        try
            if isempty(provider.api_key)
                throw(ArgumentError("api_key cannot be empty"))
            end

            url = "$(provider.base_url)/$(provider.api_version)/$endpoint?alt=sse&key=$(provider.api_key)"
            headers = Dict("Content-Type" => "application/json")

            serialized_body = isempty(body) ? UInt8[] : JSON3.write(body)

            # Use a simpler approach with HTTP.get
            response = HTTP.request(
                "POST", url, headers, serialized_body; status_exception=false
            )

            if response.status >= 400
                error_msg = String(response.body)
                put!(
                    result_channel,
                    (
                        error=ErrorException(
                            "Request failed with status $(response.status): $error_msg"
                        ),
                        text="",
                        finish_reason=nothing,
                        is_final=true,
                    ),
                )
                return nothing
            end

            # Process the streaming response
            buffer = IOBuffer()
            current_text = ""
            finish_reason = nothing

            # Split the response by data: lines
            response_text = String(response.body)
            lines = split(response_text, "\n")

            for line in lines
                # Skip empty lines and SSE prefixes
                if isempty(line) || startswith(line, ":")
                    continue
                end

                # Extract the data part (SSE format: "data: {json}")
                if !startswith(line, "data:")
                    continue
                end

                data_str = strip(replace(line, r"^data:" => ""))

                # Skip empty data or end marker
                if isempty(data_str) || data_str == "[DONE]"
                    continue
                end

                # Parse the JSON data
                try
                    parsed_data = JSON3.read(data_str)

                    # Check if there are candidates
                    if haskey(parsed_data, :candidates) && !isempty(parsed_data.candidates)
                        candidate = parsed_data.candidates[1]

                        # Extract text from the candidate
                        if haskey(candidate, :content) && haskey(candidate.content, :parts)
                            chunk_text = ""
                            for part in candidate.content.parts
                                if haskey(part, :text)
                                    chunk_text *= part.text
                                end
                            end

                            # Update the current text with the new chunk
                            current_text = chunk_text

                            # Check for finish reason
                            is_final = false
                            if haskey(candidate, :finishReason) &&
                                candidate.finishReason != "FINISH_REASON_UNSPECIFIED" &&
                                candidate.finishReason != ""
                                finish_reason = candidate.finishReason
                                is_final = true
                            end

                            # Put the current chunk into the channel
                            put!(
                                result_channel,
                                (
                                    text=current_text,
                                    finish_reason=finish_reason,
                                    is_final=is_final,
                                ),
                            )

                            # If this is the final chunk, we're done
                            if is_final
                                break
                            end
                        end
                    end
                catch e
                    # Skip malformed JSON
                    @warn "Error parsing SSE data: $e"
                    continue
                end
            end

            # Ensure we always send a final chunk if we haven't already
            if !isempty(current_text) && finish_reason === nothing
                put!(
                    result_channel, (text=current_text, finish_reason="STOP", is_final=true)
                )
            end
        catch e
            # Put the error in the channel
            put!(result_channel, (error=e, text="", finish_reason=nothing, is_final=true))
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
