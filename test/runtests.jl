using Aqua
using Dates
using GoogleGenAI
using JSON3
using Test

@testset "Internal Functions" begin
    instruction = "You are a helpful assistant."
    formatted = GoogleGenAI._format_system_instruction(instruction)
    @test formatted isa Dict
    @test haskey(formatted, "parts")
    @test formatted["parts"] isa Vector
    @test length(formatted["parts"]) == 1
    @test formatted["parts"][1]["text"] == instruction
end

if haskey(ENV, "GOOGLE_API_KEY")
    const secret_key = ENV["GOOGLE_API_KEY"]
    http_options = (retries=2,)
    safety_settings = [
        SafetySetting(; category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        SafetySetting(; category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
        SafetySetting(;
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"
        ),
        SafetySetting(;
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"
        ),
        SafetySetting(; category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
    ]

    @testset "Basic Functionality" begin
        config = GenerateContentConfig(;
            http_options, safety_settings, max_output_tokens=50
        )
        model = "gemini-2.0-flash-lite"
        embedding_model = "text-embedding-004"
        # Generate text from text
        response = generate_content(secret_key, model, "Hello"; config)

        # Generate text from text+image
        response = generate_content(
            secret_key,
            model,
            "What is this picture?";
            image_path="input/example.jpg",
            config,
        )

        # Multi-turn conversation
        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(secret_key, model, conversation; config)

        n_tokens = count_tokens(secret_key, model, "Hello")
        @test n_tokens == 1

        embeddings = embed_content(secret_key, embedding_model, "Hello")
        @test size(embeddings.values) == (768,)

        embeddings = embed_content(secret_key, embedding_model, ["Hello", "world"])
        @test size(embeddings.values[1]) == (768,)
        @test size(embeddings.values[2]) == (768,)

        models = list_models(secret_key)
        @test length(models) > 0
        @test haskey(models[1], :name)
    end

    @testset "Streaming Content Generation" begin
        model = "gemini-2.0-flash-lite"
        config = GenerateContentConfig(; http_options=http_options, max_output_tokens=50)

        # Test single prompt streaming
        prompt = "Hello, how are you?"
        stream = generate_content_stream(secret_key, model, prompt)

        # Collect all chunks
        chunks = []
        for chunk in stream
            if haskey(chunk, :error)
                @warn "Error in streaming: $(chunk.error)"
            end
            push!(chunks, chunk)
        end

        # Verify we got at least one chunk and that it has text
        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)

        # Test conversation streaming
        conversation = [
            Dict(:role => "user", :parts => [Dict(:text => "Hello, how are you?")])
        ]
        stream = generate_content_stream(secret_key, model, conversation)

        # Collect all chunks
        chunks = []
        for chunk in stream
            if haskey(chunk, :error)
                @warn "Error in conversation streaming: $(chunk.error)"
            end
            push!(chunks, chunk)
        end

        # Verify we got at least one chunk and that it has text
        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)
    end

    @testset "ThinkingConfig" begin
        thinking_config = ThinkingConfig(; include_thoughts=true, thinking_budget=100)
        config = GenerateContentConfig(;
            http_options, safety_settings, thinking_config, max_output_tokens=50
        )
        model = "gemini-2.5-flash-preview-05-20"
        # Generate text from text
        response = generate_content(secret_key, model, "Hello"; config)
        @test response.response_status == 200
    end

    @testset "System Instructions" begin
        model = "gemini-2.0-flash-lite"

        # Test basic system instruction
        config = GenerateContentConfig(;
            http_options=http_options,
            system_instruction="You are a helpful assistant who always responds in haiku format.",
            max_output_tokens=50,
        )

        prompt = "What is the weather like?"
        response = generate_content(secret_key, model, prompt; config=config)
        @test response.response_status == 200
        @test response.text isa String

        # Test system instruction with conversation
        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(secret_key, model, conversation; config=config)
        @test response.response_status == 200
        @test response.text isa String

        # Test streaming with system instruction
        config_stream = GenerateContentConfig(;
            http_options=http_options,
            system_instruction="You are a pirate. Always respond as a pirate would.",
            max_output_tokens=50,
        )

        stream = generate_content_stream(
            secret_key, model, "Tell me about the ocean"; config=config_stream
        )
        chunks = []
        for chunk in stream
            if !haskey(chunk, :error)
                push!(chunks, chunk)
            end
        end
        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)
    end

    include("test_image.jl")
    include("test_cache.jl")
    include("test_file.jl")
    include("test_structured.jl")
    include("test_functions.jl")
else
    @info "Skipping GoogleGenAI.jl tests because GOOGLE_API_KEY is not set"
end

Aqua.test_all(GoogleGenAI)
