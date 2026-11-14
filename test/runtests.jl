using Aqua
using Base64
using Dates
using GoogleGenAI
using HTTP
using JSON3
using Sockets
using Test

# Serves `handler(req)` on a free local port and passes a provider pointing at it to `f`.
function with_server(f, handler; provider_kwargs...)
    port, socket = Sockets.listenany(Sockets.localhost, 18755)
    server = HTTP.serve!(handler, socket)
    try
        f(
            GoogleProvider(;
                api_key="test", base_url="http://127.0.0.1:$port", provider_kwargs...
            ),
        )
    finally
        close(server)
    end
end

function json_response(body; status=200, headers=[])
    return HTTP.Response(status, headers; body=JSON3.write(body))
end

@testset "System instruction forms" begin
    instruction = "You are a helpful assistant."
    @test GoogleGenAI._to_content(instruction, nothing) ==
        Dict(:parts => [Dict(:text => instruction)])
    @test GoogleGenAI._to_content(["Be brief.", "Use metric units."], nothing)[:parts] ==
        [Dict(:text => "Be brief."), Dict(:text => "Use metric units.")]
    content = Dict(:parts => [Dict(:text => "raw")])
    @test GoogleGenAI._to_content(content, nothing) === content

    config = GenerateContentConfig(; system_instruction=["Be brief.", "Be kind."])
    body = GoogleGenAI._build_request_body(GoogleGenAI._to_contents("hi"), config)
    @test length(body["systemInstruction"][:parts]) == 2
    @test !haskey(body["systemInstruction"], :role)
end

@testset "Logprobs Generation Config" begin
    config = GenerateContentConfig(; response_logprobs=true, logprobs=5)
    generation_config = GoogleGenAI._build_generation_config(config)

    @test generation_config["responseLogprobs"] === true
    @test generation_config["logprobs"] == 5
end

@testset "Logprobs Defaults Are Omitted" begin
    generation_config = GoogleGenAI._build_generation_config(GenerateContentConfig())

    @test !haskey(generation_config, "responseLogprobs")
    @test !haskey(generation_config, "logprobs")
end

# Minimal stand-in for an HTTP.jl response so _parse_response can be tested
# offline against saved payloads.
struct FakeResponse
    body::Vector{UInt8}
    status::Int
end
function FakeResponse(body::AbstractString; status=200)
    return FakeResponse(Vector{UInt8}(codeunits(body)), status)
end

load_fixture(name) = read(joinpath(@__DIR__, "fixtures", name), String)

@testset "Response Parsing" begin
    response = GoogleGenAI._parse_response(FakeResponse(load_fixture("generate.json")))

    @test response.text == "Hello"
    @test response.response_status == 200
    @test response.finish_reason == "STOP"
    @test response.images isa Vector
    @test isempty(response.images)
    @test response.function_calls === nothing
    @test response.usage_metadata[:totalTokenCount] == 11
end

@testset "Response Parsing: Function Calls" begin
    response = GoogleGenAI._parse_response(FakeResponse(load_fixture("function_call.json")))

    @test response.function_calls !== nothing
    fc = only(response.function_calls)
    @test fc isa FunctionCall
    @test fc.name == "get_current_weather"
    @test fc.args["location"] == "Tokyo"
end

@testset "Error Payload Parsing" begin
    response = GoogleGenAI._parse_response(FakeResponse(load_fixture("error_429.json")))
    @test isempty(response.text)
    @test isempty(response.candidates)
end

@testset "GeminiAPIError" begin
    err = GeminiAPIError(HTTP.Response(429, []; body=load_fixture("error_429.json")))
    @test err.code == 429
    @test err.status == "RESOURCE_EXHAUSTED"
    @test startswith(err.message, "You exceeded your current quota")
    @test length(err.details) == 2
    @test err.retry_delay == 37.0
    @test sprint(showerror, err) == "GeminiAPIError: 429 RESOURCE_EXHAUSTED: $(err.message)"

    header = GeminiAPIError(HTTP.Response(503, ["Retry-After" => "2"]; body="overloaded"))
    @test (header.code, header.status, header.message) == (503, "", "overloaded")
    @test header.retry_delay == 2.0

    listed = GeminiAPIError(400, """[{"error": {"code": 400, "message": "bad"}}]""")
    @test listed.message == "bad"

    named = GeminiAPIError(
        400, """{"error": {"code": "invalid_request", "message": "bad"}}"""
    )
    @test (named.code, named.status, named.message) == (400, "invalid_request", "bad")
end

@testset "Errors are thrown as GeminiAPIError" begin
    with_server(_ -> HTTP.Response(429, load_fixture("error_429.json"))) do provider
        err = try
            generate_content(provider, "m", "hi")
        catch e
            e
        end
        @test err isa GeminiAPIError
        @test err.status == "RESOURCE_EXHAUSTED"
        @test !occursin("test", sprint(showerror, err))
    end
end

@testset "Retries follow RetryOptions" begin
    statuses = [503, 429, 200]
    seen = Int[]
    handler = function (req)
        status = statuses[length(seen) + 1]
        push!(seen, status)
        status == 200 && return json_response(Dict("totalTokens" => 3))
        status == 429 && return HTTP.Response(429, load_fixture("error_429.json"))
        return json_response(
            Dict("error" => Dict("code" => status, "message" => "no")); status
        )
    end
    options = RetryOptions(; initial_delay=0.0, jitter=0.0, max_delay=0.01)

    with_server(handler; retry_options=options) do provider
        @test count_tokens(provider, "m", "hi") == 3
    end
    @test seen == [503, 429, 200]

    empty!(seen)
    with_server(handler) do provider
        @test_throws GeminiAPIError count_tokens(provider, "m", "hi")
    end
    @test seen == [503]

    empty!(seen)
    statuses = [400, 200]
    with_server(handler; retry_options=options) do provider
        @test_throws GeminiAPIError count_tokens(provider, "m", "hi")
    end
    @test seen == [400]

    empty!(seen)
    statuses = [503, 503, 503]
    with_server(
        handler; retry_options=RetryOptions(; attempts=2, initial_delay=0.0, jitter=0.0)
    ) do provider
        @test_throws GeminiAPIError count_tokens(provider, "m", "hi")
    end
    @test seen == [503, 503]

    backoff = RetryOptions(; initial_delay=1.0, exp_base=2.0, jitter=0.0)
    @test GoogleGenAI._retry_delay(backoff, 3, GeminiAPIError(503, "")) == 4.0
    @test GoogleGenAI._retry_delay(
        RetryOptions(; max_delay=5.0), 1, GeminiAPIError(429, "", 37.0)
    ) == 5.0
    @test GoogleGenAI._retry_delay(
        RetryOptions(; initial_delay=10.0, max_delay=5.0), 1, GeminiAPIError(503, "")
    ) == 5.0
end

@testset "Streaming retries before the first chunk" begin
    calls = Ref(0)
    handler = function (http::HTTP.Stream)
        read(http)
        calls[] += 1
        if calls[] == 1
            HTTP.setstatus(http, 503)
            HTTP.startwrite(http)
            write(http, """{"error": {"code": 503, "message": "busy"}}""")
            return nothing
        end
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        event = Dict(
            "candidates" => [
                Dict(
                    "content" => Dict("parts" => [Dict("text" => "ok")]),
                    "finishReason" => "STOP",
                ),
            ],
        )
        write(http, "data: $(JSON3.write(event))\r\n\r\n")
        return nothing
    end
    port, socket = Sockets.listenany(Sockets.localhost, 18756)
    server = HTTP.serve!(handler, socket; stream=true)
    try
        options = RetryOptions(; initial_delay=0.0, jitter=0.0)
        provider = GoogleProvider(;
            api_key="test", base_url="http://127.0.0.1:$port", retry_options=options
        )
        chunks = collect(generate_content_stream(provider, "m", "hi"))
        @test [c.text for c in chunks] == ["ok"]
        @test calls[] == 2
    finally
        close(server)
    end
end

@testset "Provider http_options apply to every request" begin
    provider = GoogleProvider(; api_key="k", http_options=(readtimeout=5,))
    @test provider.http_options == (readtimeout=5,)
    @test GoogleProvider("k", "u", "v").retry_options === nothing

    slow = function (req)
        sleep(2)
        return json_response(Dict("totalTokens" => 1))
    end
    with_server(slow; http_options=(readtimeout=1, retry=false)) do provider
        @test_throws HTTP.Exceptions.TimeoutError count_tokens(provider, "m", "hi")
        patient = GenerateContentConfig(; http_options=(readtimeout=10,))
        @test count_tokens(provider, "m", "hi"; config=patient) == 1
    end
end

struct BareProvider <: GoogleGenAI.AbstractGoogleProvider
    api_key::String
    base_url::String
    api_version::String
end

@testset "Custom provider types still work" begin
    with_server(_ -> json_response(Dict("totalTokens" => 2))) do provider
        bare = BareProvider("test", provider.base_url, "v1beta")
        @test count_tokens(bare, "m", "hi") == 2
    end
end

@testset "Listing fetches every page" begin
    pages = Dict(
        "" => Dict("files" => [Dict("name" => "files/a")], "nextPageToken" => "p2"),
        "p2" => Dict("files" => [Dict("name" => "files/b")], "nextPageToken" => "p3"),
        "p3" => Dict(),
    )
    targets = String[]
    handler = function (req)
        push!(targets, req.target)
        query = HTTP.queryparams(HTTP.URI(req.target))
        return json_response(pages[get(query, "pageToken", "")])
    end
    with_server(handler) do provider
        @test [f.name for f in list_files(provider)] == ["files/a", "files/b"]
    end
    @test targets == [
        "/v1beta/files?pageSize=100",
        "/v1beta/files?pageSize=100&pageToken=p2",
        "/v1beta/files?pageSize=100&pageToken=p3",
    ]

    with_server(_ -> json_response(Dict())) do provider
        @test isempty(list_files(provider))
        @test isempty(list_cached_content(provider))
        @test isempty(list_batch_jobs(provider))
        @test isempty(list_models(provider))
    end
end

@testset "Models" begin
    model = Dict(
        "name" => "models/gemini-x",
        "displayName" => "Gemini X",
        "inputTokenLimit" => 100,
        "supportedGenerationMethods" => ["generateContent"],
    )
    targets = String[]
    handler = function (req)
        push!(targets, req.target)
        endswith(req.target, "gemini-x") && return json_response(model)
        return json_response(Dict("models" => [model]))
    end
    with_server(handler) do provider
        listed = only(list_models(provider))
        @test listed[:name] == "gemini-x"
        @test listed[:display_name] == "Gemini X"
        @test listed[:version] === nothing
        @test get_model(provider, "gemini-x") == listed
        @test get_model(provider, "models/gemini-x") == listed
    end
    @test targets == [
        "/v1beta/models?pageSize=1000", "/v1beta/models/gemini-x", "/v1beta/models/gemini-x"
    ]
end

@testset "count_tokens sends the whole request" begin
    bodies = []
    handler = function (req)
        push!(bodies, JSON3.read(req.body))
        return json_response(Dict("totalTokens" => 7))
    end
    config = GenerateContentConfig(;
        system_instruction="Be a pirate.", tools=[google_search_tool()]
    )
    with_server(handler) do provider
        @test count_tokens(provider, "m", "hi") == 7
        @test count_tokens(provider, "m", ["a", "b"]; config) == 7
    end
    request = bodies[2].generateContentRequest
    @test request.model == "models/m"
    @test [p.text for p in request.contents[1].parts] == ["a", "b"]
    @test request.systemInstruction.parts[1].text == "Be a pirate."
    @test haskey(request.tools[1], :googleSearch)
    @test !haskey(bodies[1].generateContentRequest, :generationConfig)
end

@testset "Embedding options and inputs" begin
    bodies = []
    handler = function (req)
        body = JSON3.read(req.body)
        push!(bodies, body)
        embeddings = [Dict("values" => [0.5, 1]) for _ in body.requests]
        return json_response(Dict("embeddings" => embeddings))
    end
    with_server(handler) do provider
        single = embed_content(
            provider,
            "e",
            "Hello";
            task_type="RETRIEVAL_DOCUMENT",
            title="Doc",
            output_dimensionality=2,
        )
        @test single.values == [0.5, 1.0]
        @test single.values isa Vector{Float64}
        batch = embed_content(provider, "e", ["a", (data=UInt8[1], mime_type="image/png")])
        @test length(batch.values) == 2
    end
    request = only(bodies[1].requests)
    @test request.model == "models/e"
    @test request.content.parts[1].text == "Hello"
    @test (request.taskType, request.title, request.outputDimensionality) ==
        ("RETRIEVAL_DOCUMENT", "Doc", 2)
    @test bodies[2].requests[2].content.parts[1].inlineData.mimeType == "image/png"
    @test !haskey(bodies[2].requests[1], :taskType)
end

@testset "Cache create and update bodies" begin
    bodies = []
    handler = function (req)
        push!(bodies, (method=req.method, body=JSON3.read(req.body)))
        return json_response(Dict("name" => "cachedContents/c"))
    end
    with_server(handler) do provider
        create_cached_content(provider, "m", "long text")
        create_cached_content(
            provider,
            "m",
            ["doc", (uri="https://x/f", mime_type="application/pdf")];
            expire_time=DateTime(2030, 1, 2, 3, 4, 5),
            display_name="docs",
            system_instruction=["Be brief."],
            tools=[url_context_tool()],
            tool_config=ToolConfig(;
                function_calling_config=FunctionCallingConfig(; mode="NONE")
            ),
        )
        update_cached_content(provider, "cachedContents/c", "600s")
        update_cached_content(
            provider, "cachedContents/c"; expire_time="2030-01-01T00:00:00Z"
        )
        @test_throws ArgumentError create_cached_content(
            provider, "m", "x"; ttl="1s", expire_time="t"
        )
        @test_throws ArgumentError update_cached_content(provider, "cachedContents/c")
    end
    plain, full, ttl, expire = bodies
    @test plain.body.ttl == "300s"
    @test plain.body.model == "models/m"
    @test plain.body.contents[1].parts[1].text == "long text"
    @test !haskey(plain.body, :tools)
    @test !haskey(full.body, :ttl)
    @test full.body.expireTime == "2030-01-02T03:04:05.000Z"
    @test full.body.displayName == "docs"
    @test full.body.contents[1].parts[2].fileData.fileUri == "https://x/f"
    @test full.body.systemInstruction.parts[1].text == "Be brief."
    @test haskey(full.body.tools[1], :urlContext)
    @test full.body.toolConfig.functionCallingConfig.mode == "NONE"
    @test (ttl.method, ttl.body.ttl) == ("PATCH", "600s")
    @test expire.body.expireTime == "2030-01-01T00:00:00Z"
end

@testset "Bare function tools build declarations" begin
    my_tool(x) = x
    config = GenerateContentConfig(; tools=[my_tool])
    conversation = [Dict(:role => "user", :parts => [Dict(:text => "hi")])]
    body = GoogleGenAI._build_request_body(conversation, config)
    decls = body["tools"][1]["functionDeclarations"]
    @test length(decls) == 1
    @test decls[1]["name"] == "my_tool"
    @test haskey(decls[1], "parameters")
end

@testset "Generation config sends every set field" begin
    config = GenerateContentConfig(;
        seed=42, presence_penalty=0.5, frequency_penalty=0.3, labels=Dict("team" => "a")
    )
    conversation = [Dict{Symbol,Any}(:role => "user", :parts => [Dict(:text => "hi")])]
    body = GoogleGenAI._build_request_body(conversation, config)

    @test body["generationConfig"] ==
        Dict("seed" => 42, "presencePenalty" => 0.5, "frequencyPenalty" => 0.3)
    @test body["labels"] == Dict("team" => "a")
end

@testset "Requests send the API key in a header and use the right paths" begin
    requests = []
    handler = function (req)
        push!(
            requests,
            (method=req.method, target=req.target, key=HTTP.header(req, "x-goog-api-key")),
        )
        return json_response(Dict())
    end
    with_server(handler) do provider
        delete_batch_job(provider, "batches/123")
        list_files(provider; page_size=5)
        download_batch_output_file(provider, "files/out")
        @test download_file(provider, "files/out") == codeunits("{}")
        path = tempname()
        @test download_file(provider, "files/out"; path) == path
        @test read(path, String) == "{}"
    end

    @test [(r.method, r.target) for r in requests] == [
        ("DELETE", "/v1beta/batches/123"),
        ("GET", "/v1beta/files?pageSize=5"),
        ("GET", "/download/v1beta/files/out:download?alt=media"),
        ("GET", "/download/v1beta/files/out:download?alt=media"),
        ("GET", "/download/v1beta/files/out:download?alt=media"),
    ]
    @test all(r -> r.key == "test", requests)
end

include("test_parsing.jl")
include("test_builders.jl")
include("test_chat.jl")
include("test_batch.jl")
include("test_live.jl")
include("test_stream.jl")
include("test_decisions.jl")

# Live API integration tests. Opt in with:
#   GOOGLE_GENAI_INTEGRATION=1 julia --project=. -e 'using Pkg; Pkg.test()'
include("integration.jl")

Aqua.test_all(GoogleGenAI)
