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
            :parts =>
                [Dict(:text => "We previously discussed some text. Summarize it again.")],
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
    sleep(5)
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
