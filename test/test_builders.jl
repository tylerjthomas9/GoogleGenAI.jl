@testset "Generation Config Mapping" begin
    config = GenerateContentConfig(;
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        candidate_count=2,
        max_output_tokens=256,
        stop_sequences=["END"],
        response_mime_type="application/json",
    )
    gc = GoogleGenAI._build_generation_config(config)

    @test gc["temperature"] == 0.5
    @test gc["topP"] == 0.9
    @test gc["topK"] == 40
    @test gc["candidateCount"] == 2
    @test gc["maxOutputTokens"] == 256
    @test gc["stopSequences"] == ["END"]
    @test gc["responseMimeType"] == "application/json"
end

@testset "Generation Config: ThinkingConfig" begin
    gc(tc) =
        GoogleGenAI._build_generation_config(GenerateContentConfig(; thinking_config=tc))

    @test gc(ThinkingConfig(; include_thoughts=true, thinking_budget=512))["thinkingConfig"] ==
        Dict("includeThoughts" => true, "thinkingBudget" => 512)
    @test gc(ThinkingConfig(; thinking_level="low"))["thinkingConfig"] ==
        Dict("thinkingLevel" => "LOW")
    @test gc(ThinkingConfig())["thinkingConfig"] == Dict()
    @test ThinkingConfig(; thinking_level="XHigh").thinking_level == "XHIGH"
    @test_throws ArgumentError ThinkingConfig(; thinking_level="extreme")
    @test_throws ArgumentError ThinkingConfig(; thinking_level="low", thinking_budget=10)
end

@testset "Generation Config: newer fields" begin
    schema = Dict("type" => "object", "properties" => Dict("a" => Dict("type" => "string")))
    config = GenerateContentConfig(;
        response_mime_type="application/json",
        response_json_schema=schema,
        image_config=Dict(:aspectRatio => "16:9"),
        service_tier="flex",
    )
    body = GoogleGenAI._build_request_body(GoogleGenAI._to_contents("hi"), config)
    @test body["generationConfig"]["responseJsonSchema"] === schema
    @test body["generationConfig"]["imageConfig"] == Dict(:aspectRatio => "16:9")
    @test body["serviceTier"] == "flex"
    @test !haskey(body["generationConfig"], "serviceTier")
    @test GenerateContentConfig(; response_schema=schema).response_schema === schema

    minimal = GoogleGenAI._build_request_body(
        GoogleGenAI._to_contents("hi"), GenerateContentConfig()
    )
    @test collect(keys(minimal)) == ["contents"]
end
@testset "Function Declaration Conversion" begin
    decl = FunctionDeclaration(;
        name="get_weather",
        description="Get current weather",
        parameters=FunctionParameter(;
            type="object",
            properties=Dict{String,Any}("location" => Dict{String,Any}("type" => "string")),
            required=["location"],
        ),
    )
    api = GoogleGenAI.to_api_function_declaration(decl)

    @test api["name"] == "get_weather"
    @test api["description"] == "Get current weather"
    @test api["parameters"]["type"] == "object"
    @test haskey(api["parameters"]["properties"], "location")
    @test api["parameters"]["required"] == ["location"]
end

@testset "SafetySetting Validation" begin
    @test SafetySetting("HARM_CATEGORY_HARASSMENT", "BLOCK_NONE") isa SafetySetting
    @test_throws ArgumentError SafetySetting("NOT_A_CATEGORY", "BLOCK_NONE")
    @test_throws ArgumentError SafetySetting("HARM_CATEGORY_HARASSMENT", "NOT_A_THRESHOLD")
end

@testset "Native Tool Detection" begin
    @test is_native_tool(Dict{Symbol,Any}(:googleSearch => Dict())) == (true, GOOGLE_SEARCH)
    @test is_native_tool(Dict{Symbol,Any}(:codeExecution => Dict())) ==
        (true, CODE_EXECUTION)
    @test is_native_tool(Dict("functionDeclarations" => [])) == (true, FUNCTION_CALLING)
    @test is_native_tool(Dict(:mcpServers => [])) == (true, MCP_SERVERS)
    @test is_native_tool(Dict{Symbol,Any}(:unknownTool => Dict())) == (false, nothing)
end

@testset "Built-in tool helpers" begin
    @test google_search_tool() == Dict("googleSearch" => Dict())
    @test code_execution_tool() == Dict("codeExecution" => Dict())
    @test url_context_tool() == Dict("urlContext" => Dict())
    @test google_maps_tool() == Dict("googleMaps" => Dict())
    @test google_maps_tool(; enable_widget=true) ==
        Dict("googleMaps" => Dict("enableWidget" => true))
    @test file_search_tool(["fileSearchStores/a"]; top_k=3) == Dict(
        "fileSearch" => Dict("fileSearchStoreNames" => ["fileSearchStores/a"], "topK" => 3)
    )
    @test computer_use_tool() ==
        Dict("computerUse" => Dict("environment" => "ENVIRONMENT_BROWSER"))

    tools = [
        google_search_tool(),
        url_context_tool(),
        google_maps_tool(),
        file_search_tool(["s"]),
        computer_use_tool(),
    ]
    @test [last(is_native_tool(t)) for t in tools] == [GOOGLE_SEARCH, URL_CONTEXT, GOOGLE_MAPS, FILE_SEARCH, COMPUTER_USE]

    decl = FunctionDeclaration(; name="f", parameters=FunctionParameter(; type="object"))
    built = GoogleGenAI._build_tools([url_context_tool(), decl])
    @test built == [
        Dict("urlContext" => Dict()),
        Dict(
            "functionDeclarations" =>
                [Dict("name" => "f", "parameters" => Dict("type" => "object"))],
        ),
    ]
    @test_throws ArgumentError GoogleGenAI._build_tools(["not a tool"])
end

@testset "Contents normalization" begin
    file = JSON3.read(
        """{"name": "files/a", "uri": "https://x/files/a", "mimeType": "application/pdf"}"""
    )
    turn = only(
        GoogleGenAI._to_contents([
            "Summarize", file, (data=UInt8[1, 2], mime_type="image/png")
        ]),
    )
    @test turn[:role] == "user"
    @test turn[:parts][1] == Dict(:text => "Summarize")
    @test turn[:parts][2] == Dict(
        :fileData => Dict(:fileUri => "https://x/files/a", :mimeType => "application/pdf")
    )
    @test turn[:parts][3][:inlineData] == Dict(:mimeType => "image/png", :data => "AQI=")

    conversation = [
        Dict(:role => "user", :parts => [Dict(:text => "hi")]),
        Dict(:role => "model", :parts => [Dict(:text => "hello")]),
    ]
    @test GoogleGenAI._to_contents(conversation) === conversation

    with_image = GoogleGenAI._with_media(
        conversation[1:1], "", [(data="AAAA", mime_type="image/jpeg")]
    )
    @test length(with_image[1][:parts]) == 2
    @test length(conversation[1][:parts]) == 1
    @test GoogleGenAI._with_media("hi", "", NamedTuple[]) ==
        [Dict(:role => "user", :parts => [Dict(:text => "hi")])]

    @test GoogleGenAI._to_part(Dict("uri" => "u", "mimeType" => "text/plain")) ==
        GoogleGenAI._file_part("u", "text/plain")
    @test_throws ArgumentError GoogleGenAI._to_part((uri="https://x",))
    @test GoogleGenAI._model_path("gemini-x") == "models/gemini-x"
    @test GoogleGenAI._model_path("tunedModels/t") == "tunedModels/t"
end
