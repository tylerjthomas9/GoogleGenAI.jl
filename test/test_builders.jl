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
    tc = ThinkingConfig(; include_thoughts=true, thinking_budget=512)
    gc = GoogleGenAI._build_generation_config(GenerateContentConfig(; thinking_config=tc))

    @test gc["thinkingConfig"] === tc
end

@testset "Function Declaration Conversion" begin
    decl = FunctionDeclaration(;
        name="get_weather",
        description="Get current weather",
        parameters=FunctionParameter(;
            type="object",
            properties=Dict{String,Any}(
                "location" => Dict{String,Any}("type" => "string"),
            ),
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
    @test GoogleGenAI.is_native_tool(Dict{Symbol,Any}(:googleSearch => Dict())) ==
          (true, GOOGLE_SEARCH)
    @test GoogleGenAI.is_native_tool(Dict{Symbol,Any}(:codeExecution => Dict())) ==
          (true, CODE_EXECUTION)
    custom = Dict{Symbol,Any}(:unknownTool => Dict{String,Any}())
    @test GoogleGenAI.is_native_tool(custom) == (false, nothing)
end
