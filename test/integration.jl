# Live API integration tests.
#
# These hit the real Gemini API and are opt-in:
#
#   GOOGLE_GENAI_INTEGRATION=1 julia --project=. -e 'using Pkg; Pkg.test()'
#
# They are skipped by default so that `Pkg.test()` is fast, free,
# deterministic, and runnable without an API key.

if get(ENV, "GOOGLE_GENAI_INTEGRATION", "0") != "1"
    @info "Skipping integration tests (set GOOGLE_GENAI_INTEGRATION=1 to run live API tests)"
elseif !haskey(ENV, "GOOGLE_API_KEY") && !haskey(ENV, "GEMINI_API_KEY")
    @info "GOOGLE_GENAI_INTEGRATION=1 but GOOGLE_API_KEY/GEMINI_API_KEY not set; skipping"
else
    if haskey(ENV, "GOOGLE_API_KEY")
        @info "Using GOOGLE_API_KEY"
        const secret_key = ENV["GOOGLE_API_KEY"]
    else
        @info "Using GEMINI_API_KEY"
        const secret_key = ENV["GEMINI_API_KEY"]
    end
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
        model = "gemini-3.5-flash-lite"
        embedding_model = "gemini-embedding-001"
        response = generate_content(secret_key, model, "Hello"; config)

        response = generate_content(
            secret_key,
            model,
            "What is this picture?";
            image_path="input/example.jpg",
            config,
        )

        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(secret_key, model, conversation; config)

        n_tokens = count_tokens(secret_key, model, "Hello")
        @test n_tokens == 1

        embeddings = embed_content(secret_key, embedding_model, "Hello")
        @test size(embeddings.values) == (3072,)

        embeddings = embed_content(secret_key, embedding_model, ["Hello", "world"])
        @test size(embeddings.values[1]) == (3072,)
        @test size(embeddings.values[2]) == (3072,)

        models = list_models(secret_key)
        @test length(models) > 0
        @test haskey(models[1], :name)
    end

    @testset "Streaming Content Generation" begin
        model = "gemini-3.5-flash-lite"
        config = GenerateContentConfig(; http_options=http_options, max_output_tokens=50)

        prompt = "Hello, how are you?"
        stream = generate_content_stream(secret_key, model, prompt)

        chunks = []
        for chunk in stream
            if haskey(chunk, :error)
                @warn "Error in streaming: $(chunk.error)"
            end
            push!(chunks, chunk)
        end

        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)

        conversation = [
            Dict(:role => "user", :parts => [Dict(:text => "Hello, how are you?")])
        ]
        stream = generate_content_stream(secret_key, model, conversation)

        chunks = []
        for chunk in stream
            if haskey(chunk, :error)
                @warn "Error in conversation streaming: $(chunk.error)"
            end
            push!(chunks, chunk)
        end

        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)
    end

    @testset "ThinkingConfig" begin
        thinking_config = ThinkingConfig(; include_thoughts=true, thinking_budget=100)
        config = GenerateContentConfig(;
            http_options, safety_settings, thinking_config, max_output_tokens=50
        )
        model = "gemini-3.5-flash-lite"
        response = generate_content(secret_key, model, "Hello"; config)
        @test response.response_status == 200
    end

    @testset "System Instructions" begin
        model = "gemini-3.5-flash-lite"

        config = GenerateContentConfig(;
            http_options=http_options,
            system_instruction="You are a helpful assistant who always responds in haiku format.",
            max_output_tokens=50,
        )

        prompt = "What is the weather like?"
        response = generate_content(secret_key, model, prompt; config=config)
        @test response.response_status == 200
        @test response.text isa String

        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(secret_key, model, conversation; config=config)
        @test response.response_status == 200
        @test response.text isa String

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

    @testset "Multi-Image Input" begin
        model = "gemini-3.5-flash-lite"
        prompt = "Describe the contents of these two images."
        images = [(path="input/example.jpg",), (path="input/example.jpg",)]

        response = generate_content(secret_key, model, prompt; images)
        @test response.response_status == 200
        @test !isempty(response.text)

        config = GenerateContentConfig(; http_options=http_options, max_output_tokens=50)
        stream = generate_content_stream(secret_key, model, prompt; images, config)
        chunks = []
        for chunk in stream
            if haskey(chunk, :error)
                @warn "Error in streaming: $(chunk.error)"
            end
            push!(chunks, chunk)
        end

        @test length(chunks) > 0
        @test any(chunk -> !isempty(get(chunk, :text, "")), chunks)
    end

    include("test_image.jl")
    include("test_cache.jl")
    include("test_file.jl")
    include("test_structured.jl")
    include("test_functions.jl")

    @testset "Live API (text in, audio out)" begin
        # Current Live models are audio-out only; declarations must be passed
        # via config for mid-session tool use.
        session = connect_live(
            secret_key,
            "gemini-2.5-flash-native-audio-latest";
            config=GenerateContentConfig(; response_modalities=["AUDIO"]),
        )
        try
            send_client_content(session; turns="Say exactly: PONG", turn_complete=true)
            got_audio = false
            complete = false
            while !complete
                event = recv_event(session)
                haskey(event, :audio) && (got_audio = true)
                sc = get(event, :serverContent, nothing)
                sc !== nothing && get(sc, :turnComplete, false) && (complete = true)
            end
            @test got_audio
        finally
            close_live!(session)
        end
    end

    @testset "Decisions" begin
        provider = GoogleProvider(; api_key=secret_key)
        model = "gemini-decision-1"

        r = create_decision(
            provider,
            model,
            "Third late order and support hung up on me. Cancel my subscription.",
            (
                wants_cancel=NoulQuestion("Does the customer want to cancel?"),
                topic=ChoiceQuestion("Primary topic", ["shipping", "billing", "praise"]),
                sentiment=ScoreQuestion(
                    "Customer sentiment", ["negative", "neutral", "positive"]
                ),
            );
            http_kwargs=http_options,
        )
        @test r.response_status == 200
        @test r.answers["wants_cancel"].noul > 0.5
        @test r.answers["topic"].choice == "shipping"
        sentiment = r.answers["sentiment"]
        @test sentiment.legend[argmax(sentiment.probabilities)] == "negative"
        @test r.usage[:total_output_tokens] == 0

        image = create_decision(
            secret_key,
            model,
            [(path="input/example.jpg",)],
            Dict("is_photo" => NoulQuestion("Is this a photograph?"));
            http_kwargs=http_options,
        )
        @test image.answers["is_photo"] isa NoulAnswer

        err = try
            create_decision(
                provider,
                model,
                "hi",
                (bad=Dict("type" => "choice", "instructions" => "x"),),
            )
        catch e
            e
        end
        @test err isa HTTP.Exceptions.StatusError
        @test !occursin(secret_key, sprint(showerror, err))
    end
end
