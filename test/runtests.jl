using Aqua
using Dates
using GoogleGenAI
using JSON3
using Test

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
        SafetySetting(;
            category="HARM_CATEGORY_CIVIC_INTEGRITY",
            threshold="HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        ),
    ]

    @testset "GoogleGenAI.jl" begin
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

    @testset "Image Generation" begin
        config = GenerateContentConfig(; response_modalities=["Text", "Image"])
        prompt = (
            "Hi, can you create a 3d rendered image of a pig " *
            "with wings and a top hat flying over a happy " *
            "futuristic scifi city with lots of greenery?"
        )

        response = generate_content(
            secret_key, "gemini-2.0-flash-exp-image-generation", prompt; config
        )
        @test !isempty(response.images)

        image_path = "input/example.jpg"
        model = "gemini-2.0-flash-exp-image-generation"
        prompt = "Make all of the circles green"
        response = generate_content(secret_key, model, prompt; image_path, config)
        @test !isempty(response.images)
    end

    @testset "Content Caching" begin
        model = "gemini-1.5-flash-8b"
        text = read("input/example.txt", String) * "<><"^13_860

        # 1) Create the cache
        cache_result = create_cached_content(
            secret_key,
            model,
            text;
            ttl="60s",  # cache for 60 seconds
            system_instruction="You are Julia's Number 1 fan",  # optional
        )
        cache_name = cache_result.name
        config = GenerateContentConfig(;
            http_options=http_options, max_output_tokens=50, cached_content=cache_name
        )

        # 2) Generate content with the cache (single prompt)
        single_prompt = "What is the main topic of this text?"
        response_cached = generate_content(secret_key, model, single_prompt; config)
        @test response_cached.response_status == 200

        # 3) Generate content with the cache (conversation)
        conversation = [
            Dict(
                :role => "user",
                :parts => [
                    Dict(:text => "We previously discussed some text. Summarize it again."),
                ],
            ),
        ]
        response_convo = generate_content(secret_key, model, conversation; config)
        @test response_convo.response_status == 200

        # 4) List all cached content and verify the new cache is present
        # (you won't see the actual text or tokens, just metadata)
        list_result = list_cached_content(secret_key)
        @test any(cache["name"] == cache_name for cache in list_result)

        # (Optional) You can see if our new cache_name is in the returned list:
        # (This step depends on how many caches you have in your project, but you can do something like:)
        @test any(cache["name"] == cache_name for cache in list_result)

        # 5) Get details about the newly created cache
        get_result = get_cached_content(secret_key, cache_name)
        @test get_result["name"] == cache_name

        # 6) Update the cache to extend the TTL
        sleep(2)
        update_result = update_cached_content(secret_key, cache_name, "90s")
        function parse_timestamp(ts)
            ts = replace(ts, "Z" => "")
            ts = split(ts, ".")[1]
            return DateTime(ts, dateformat"yyyy-mm-dd\THH:MM:SS")
        end
        t1 = parse_timestamp(update_result[:updateTime])
        t2 = parse_timestamp(update_result[:expireTime])
        seconds_diff = (t2 - t1).value / 1000  # Convert milliseconds to seconds
        @test abs(seconds_diff - 90) < 2

        # 7) Delete the cache
        delete_status = delete_cached_content(secret_key, cache_name)
        @test delete_status == 200  # or whatever status code you expect (200 or 204)
    end

    @testset "File Management" begin
        # test image, audio, text, video, pdf
        test_files = [
            "input/example.jpg",
            "input/example.m4a",
            "input/example.txt",
            "input/example.mp4",
            "input/example.pdf",
        ]
        uploaded_files = []
        for test_file in test_files
            # 1) Upload the file.
            upload_result = upload_file(secret_key, test_file; display_name="Test Upload")
            push!(uploaded_files, upload_result)
            @test haskey(upload_result, "name")
            file_name = upload_result["name"]

            # 2) Retrieve file metadata.
            get_result = get_file(secret_key, file_name)
            @test get_result["name"] == file_name
        end

        # Generate content with all files
        @info "Sleeping for 5 seconds to allow files to be processed"
        sleep(5) # allow time for the files to be processed 
        contents = ["What files do you have?", uploaded_files...]
        model = "gemini-2.0-flash-lite"
        response = generate_content(secret_key, model, contents;)

        # Test that all the files are available, then delete them
        list_result = list_files(secret_key; page_size=10)
        for file in uploaded_files
            @test any(f -> f["name"] == file["name"], list_result)
            delete_status = delete_file(secret_key, file["name"])
            @test delete_status == 200 || delete_status == 204
        end
    end

    @testset "Structured Generation" begin
        model = "gemini-2.0-flash-lite"
        config = GenerateContentConfig(; http_options=http_options)

        schema = Dict(
            :type => "ARRAY",
            :items => Dict(
                :type => "OBJECT",
                :properties => Dict(
                    :recipe_name => Dict(:type => "STRING"),
                    :ingredients =>
                        Dict(:type => "ARRAY", :items => Dict(:type => "STRING")),
                ),
                :propertyOrdering => ["recipe_name", "ingredients"],
            ),
        )

        config = GenerateContentConfig(;
            response_mime_type="application/json", response_schema=schema
        )

        prompt = "List a few popular cookie recipes with exact amounts of each ingredient."
        response = generate_content(secret_key, model, prompt; config=config)
        json_string = response.text
        recipes = JSON3.read(json_string)
        @test length(recipes) > 0
        @test haskey(recipes[1], "recipe_name")
        @test haskey(recipes[1], "ingredients")
    end

    @testset "Code Generation" begin
        model = "gemini-2.0-flash"
        tools = [Dict(:code_execution => Dict())]
        config = GenerateContentConfig(; http_options, tools)

        prompt = "Write a function to calculate the factorial of a number."
        response = generate_content(secret_key, model, prompt; config=config)
        @test response.response_status == 200
        @test response.text isa String
        @test occursin("```python", response.text)
    end

else
    @info "Skipping GoogleGenAI.jl tests because GOOGLE_API_KEY is not set"
end

Aqua.test_all(GoogleGenAI)
