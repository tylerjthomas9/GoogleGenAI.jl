using GoogleGenAI
using JSON3
using Test

@testset "BatchRequest payload" begin
    req = BatchRequest("Why is the sky blue?"; key="request-1")
    payload = GoogleGenAI._batch_request_payload(req)
    @test haskey(payload, "request")
    @test haskey(payload, "metadata")
    @test payload["metadata"]["key"] == "request-1"
    @test payload["request"]["contents"][1]["parts"][1]["text"] == "Why is the sky blue?"
    @test haskey(payload["request"], "generationConfig")
end

@testset "BatchRequest with config" begin
    config = GenerateContentConfig(; temperature=0.5, system_instruction="You are a cat.")
    req = BatchRequest("Write a poem."; key="poem", config=config)
    payload = GoogleGenAI._batch_request_payload(req)
    @test payload["request"]["generationConfig"]["temperature"] == 0.5
    @test payload["request"]["systemInstruction"]["parts"][1]["text"] == "You are a cat."
end

@testset "Inline requests default keys" begin
    requests = [BatchRequest("a"), BatchRequest("b"; key="custom")]
    payloads = GoogleGenAI._build_inline_requests(requests)
    @test length(payloads) == 2
    @test payloads[1]["metadata"]["key"] == "request-1"
    @test payloads[2]["metadata"]["key"] == "custom"
end

@testset "Batch request body (inline)" begin
    body = GoogleGenAI._build_batch_request_body(;
        requests=[BatchRequest("hello")], display_name="my-batch"
    )
    @test haskey(body, "batch")
    batch = body["batch"]
    @test batch["display_name"] == "my-batch"
    inner = batch["input_config"]["requests"]["requests"]
    @test length(inner) == 1
    @test haskey(inner[1], "request")

    @test_throws ArgumentError GoogleGenAI._build_batch_request_body()
    @test_throws ArgumentError GoogleGenAI._build_batch_request_body(;
        requests=[BatchRequest("x")], input_file="files/123"
    )
end

@testset "Batch request body (file input)" begin
    body = GoogleGenAI._build_batch_request_body(; input_file="files/123456")
    file_name = body["batch"]["input_config"]["requests"]["file_name"]
    @test file_name == "files/123456"
end

@testset "Completed states helper" begin
    @test GoogleGenAI._is_completed_state("JOB_STATE_SUCCEEDED")
    @test GoogleGenAI._is_completed_state("JOB_STATE_EXPIRED")
    @test !GoogleGenAI._is_completed_state("JOB_STATE_RUNNING")
    @test !GoogleGenAI._is_completed_state("JOB_STATE_PENDING")
end

@testset "extract_batch_text: inline job" begin
    job = JSON3.read("""
    {
        "name": "batches/123",
        "done": true,
        "metadata": {"state": "JOB_STATE_SUCCEEDED"},
        "response": {
            "inlinedResponses": [
                {
                    "metadata": {"key": "request-1"},
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": "Hello"}, {"text": " world"}]}}
                        ]
                    }
                },
                {
                    "metadata": {"key": "request-2"},
                    "error": {"code": 400, "message": "bad request"}
                }
            ]
        }
    }
    """)
    results = extract_batch_text(job)
    @test length(results) == 2
    @test results[1] == ("request-1" => "Hello world")
    @test results[2][1] == "request-2"
    @test startswith(results[2][2], "ERROR:")
end

@testset "extract_batch_text: JSONL result line" begin
    line = JSON3.read("""
    {
        "key": "request-1",
        "response": {
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}]
        }
    }
    """)
    results = extract_batch_text(line)
    @test results == ["request-1" => "Hi"]

    error_line = JSON3.read("""
    {"key": "request-2", "error": {"code": 429, "message": "rate limited"}}
    """)
    results = extract_batch_text(error_line)
    @test results[1][1] == "request-2"
    @test occursin("rate limited", results[1][2])
end

@testset "extract_batch_text: failed job" begin
    job = JSON3.read("""
    {
        "name": "batches/456",
        "error": {"code": 500, "message": "internal error"}
    }
    """)
    results = extract_batch_text(job)
    @test length(results) == 1
    @test occursin("internal error", results[1][2])
end

@testset "extract_batch_text: file output points caller to file" begin
    job = JSON3.read("""
    {
        "name": "batches/789",
        "response": {"responsesFile": "files/999"}
    }
    """)
    @test_throws ErrorException extract_batch_text(job)
end
