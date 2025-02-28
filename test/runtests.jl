using Aqua
using Dates
using GoogleGenAI
using Test

if haskey(ENV, "GOOGLE_API_KEY")
    const secret_key = ENV["GOOGLE_API_KEY"]

    @testset "GoogleGenAI.jl" begin
        model = "gemini-2.0-flash-lite"
        embedding_model = "text-embedding-004"
        api_kwargs = (max_output_tokens=50,)
        http_options = (retries=2,)
        config = GenerateContentConfig(; http_options=http_options)
        # Generate text from text
        response = generate_content(secret_key, model, "Hello"; api_kwargs, config)

        # Generate text from text+image
        response = generate_content(
            secret_key,
            model,
            "What is this picture?";
            image_path="example.jpg",
            api_kwargs,
            config,
        )

        # Multi-turn conversation
        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(secret_key, model, conversation; api_kwargs, config)

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

    @testset "Content Caching" begin
        model = "gemini-1.5-flash-8b"
        text = read("example.txt", String) * "<><"^13_860

        # 1) Create the cache
        cache_result = create_cached_content(
            secret_key,
            model,
            text;
            ttl="60s",  # cache for 60 seconds
            system_instruction="You are Julia's Number 1 fan",  # optional
        )
        cache_name = cache_result.name

        # 2) Generate content with the cache (single prompt)
        single_prompt = "What is the main topic of this text?"
        response_cached = generate_content_with_cache(
            secret_key, model, cache_name, single_prompt
        )
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
        response_convo = generate_content_with_cache(
            secret_key, model, cache_name, conversation
        )
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
        # Ensure the test file exists; if not, create a dummy file.
        test_file = "example.jpg"

        # 1) Upload the file.
        upload_result = upload_file(
            secret_key, test_file; display_name="Test JPEG", mime_type="image/jpeg"
        )
        @test haskey(upload_result, "name")
        file_name = upload_result["name"]

        # 2) Retrieve file metadata.
        get_result = get_file(secret_key, file_name)
        @test get_result["name"] == file_name

        # 3) List files and verify the uploaded file appears in the list.
        list_result = list_files(secret_key; page_size=10)
        @test any(f -> f["name"] == file_name, list_result)

        # 4) Delete the file.
        delete_status = delete_file(secret_key, file_name)
        @test delete_status == 200 || delete_status == 204
    end

else
    @info "Skipping GoogleGenAI.jl tests because GOOGLE_API_KEY is not set"
end

Aqua.test_all(GoogleGenAI)
