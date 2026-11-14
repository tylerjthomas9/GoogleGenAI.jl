@testset "Models and pagination" begin
    models = list_models(secret_key)
    @test length(models) > 50
    @test length(unique(m[:name] for m in models)) == length(models)
    model = get_model(secret_key, "gemini-3.5-flash-lite")
    @test model[:name] == "gemini-3.5-flash-lite"
    @test model[:prompt_token_limit] > 0
    @test any(m -> m[:name] == model[:name], models)

    err = try
        get_model(secret_key, "not-a-real-model")
    catch e
        e
    end
    @test err isa GeminiAPIError
    @test err.code == 404
end

@testset "Request options" begin
    model = "gemini-3.5-flash-lite"
    provider = GoogleProvider(;
        api_key=secret_key, retry_options=RetryOptions(), http_options=(readtimeout=120,)
    )

    plain = count_tokens(provider, model, "Hello")
    config = GenerateContentConfig(;
        system_instruction=["You are a pirate.", "Answer in rhyme."],
        tools=[url_context_tool(), google_search_tool()],
    )
    @test count_tokens(provider, model, "Hello"; config) > plain

    schema = Dict(
        "type" => "object",
        "properties" => Dict("greeting" => Dict("type" => "string")),
        "required" => ["greeting"],
    )
    config = GenerateContentConfig(;
        response_mime_type="application/json",
        response_json_schema=schema,
        service_tier="standard",
        system_instruction=Dict(:parts => [Dict(:text => "Greet in French.")]),
    )
    response = generate_content(provider, model, "Say hi"; config)
    @test haskey(JSON3.read(response.text), :greeting)

    response = generate_content(
        provider,
        model,
        ["Describe this image in five words.", (path="input/example.jpg",)];
        config=GenerateContentConfig(; max_output_tokens=50),
    )
    @test !isempty(response.text)

    config = GenerateContentConfig(; tools=[url_context_tool()], max_output_tokens=200)
    response = generate_content(
        provider,
        model,
        "What is the title of https://julialang.org ? Answer briefly.";
        config,
    )
    @test response.response_status == 200
end

@testset "Embedding options" begin
    model = "gemini-embedding-001"
    query = embed_content(
        secret_key,
        model,
        "What is Julia?";
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=256,
    )
    @test length(query.values) == 256
    docs = embed_content(
        secret_key,
        model,
        ["Julia is a programming language.", "Bananas are yellow."];
        task_type="RETRIEVAL_DOCUMENT",
        title="Facts",
        output_dimensionality=256,
    )
    @test length(docs.values) == 2
    @test all(length.(docs.values) .== 256)
end

@testset "Chat history with function calls" begin
    functions = Dict{String,Function}(
        "get_weather" => (; city) -> Dict("temperature_c" => -3, "city" => city)
    )
    weather = FunctionDeclaration(
        "get_weather",
        "Get the current weather for a city",
        Dict{String,Any}(
            "type" => "object",
            "properties" =>
                Dict{String,Any}("city" => Dict{String,Any}("type" => "string")),
            "required" => ["city"],
        ),
    )
    config = GenerateContentConfig(; function_declarations=[weather])
    chat = create_chat(secret_key, "gemini-3.5-flash-lite"; config, functions)
    response = send_message(chat, "What's the weather in Oslo right now?")
    @test occursin("3", response.text)
    roles = [turn[:role] for turn in chat.history]
    @test roles[1] == "user" && roles[end] == "model"
    @test any(turn -> any(p -> haskey(p, :functionCall), turn[:parts]), chat.history)
    @test any(turn -> any(p -> haskey(p, :functionResponse), turn[:parts]), chat.history)

    response = send_message(chat, "And what city did I ask about?")
    @test occursin("Oslo", response.text)
end

@testset "Cache options" begin
    model = "gemini-3.5-flash-lite"
    text = read("input/example.txt", String) * "<><"^13_860
    cache = create_cached_content(
        secret_key,
        model,
        text;
        expire_time=now(UTC) + Minute(5),
        display_name="parity-test",
        system_instruction=["You are Julia's Number 1 fan"],
    )
    try
        @test cache.displayName == "parity-test"
        updated = update_cached_content(
            GoogleProvider(; api_key=secret_key),
            cache.name;
            expire_time=now(UTC) + Minute(10),
        )
        @test updated.expireTime > cache.expireTime
        @test any(c -> c.name == cache.name, list_cached_content(secret_key))
    finally
        delete_cached_content(secret_key, cache.name)
    end
end

@testset "Batch jobs" begin
    provider = GoogleProvider(; api_key=secret_key)
    model = "gemini-3.5-flash-lite"
    config = GenerateContentConfig(;
        system_instruction="Answer in one word.",
        tools=[google_search_tool()],
        safety_settings=[SafetySetting("HARM_CATEGORY_HARASSMENT", "BLOCK_NONE")],
    )
    jobs = []
    try
        push!(
            jobs,
            create_batch_job(
                provider,
                model;
                requests=[BatchRequest("Capital of France?"; config)],
                display_name="parity-inline",
            ),
        )

        path = tempname() * ".jsonl"
        line = Dict(
            "key" => "q1",
            "request" => Dict("contents" => [Dict("parts" => [Dict("text" => "Hi")])]),
        )
        write(path, JSON3.write(line) * "\n")
        input = upload_file(provider, path; mime_type="application/jsonl")
        push!(jobs, create_batch_job(provider, model; input_file=input.name))

        push!(
            jobs,
            create_embeddings_batch_job(
                provider,
                "gemini-embedding-001";
                contents=["hello", "world"],
                task_type="CLUSTERING",
                display_name="parity-embed",
            ),
        )

        @test all(job -> startswith(job.name, "batches/"), jobs)
        listed = Set(job.name for job in list_batch_jobs(provider))
        @test all(job -> job.name in listed, jobs)
        delete_file(provider, input.name)
    finally
        for job in jobs
            try
                cancel_batch_job(provider, job.name)
                poll_batch_job(provider, job.name; interval_s=2, timeout_s=60)
                delete_batch_job(provider, job.name)
            catch e
                @warn "Batch cleanup failed for $(job.name)" exception = e
            end
        end
    end
end
