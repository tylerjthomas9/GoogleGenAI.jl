using Aqua
using Dates
using GoogleGenAI
using HTTP
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
    # The 429/RESOURCE_EXHAUSTED body should parse without error and yield
    # no candidates/text.
    response = GoogleGenAI._parse_response(FakeResponse(load_fixture("error_429.json")))

    @test isempty(response.text)
    @test isempty(response.candidates)
end

@testset "status_error includes response body" begin
    resp = HTTP.Response(429, []; body = "quota exceeded")
    err = try
        GoogleGenAI.status_error(resp)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("429", err.msg)
    @test occursin("quota exceeded", err.msg)
end

@testset "Bare function tools build declarations" begin
    my_tool(x) = x
    config = GenerateContentConfig(; tools = [my_tool])
    conversation = [Dict(:role => "user", :parts => [Dict(:text => "hi")])]
    body = GoogleGenAI._build_request_body(conversation, config)
    decls = body["tools"][1]["functionDeclarations"]
    @test length(decls) == 1
    @test decls[1]["name"] == "my_tool"
    @test haskey(decls[1], "parameters")
end

include("test_parsing.jl")
include("test_builders.jl")
include("test_chat.jl")
include("test_batch.jl")
include("test_live.jl")

# Live API integration tests. Opt in with:
#   GOOGLE_GENAI_INTEGRATION=1 julia --project=. -e 'using Pkg; Pkg.test()'
include("integration.jl")

Aqua.test_all(GoogleGenAI)
