using GoogleGenAI
using JSON3
using Test

@testset "BatchRequest payload" begin
    req = BatchRequest("Why is the sky blue?"; key="request-1")
    payload = GoogleGenAI._batch_request_payload(req)
    @test haskey(payload, "request")
    @test haskey(payload, "metadata")
    @test payload["metadata"]["key"] == "request-1"
    @test payload["request"]["contents"][1][:parts][1][:text] == "Why is the sky blue?"
end

@testset "BatchRequest with config" begin
    config = GenerateContentConfig(; temperature=0.5, system_instruction="You are a cat.")
    req = BatchRequest("Write a poem."; key="poem", config=config)
    payload = GoogleGenAI._batch_request_payload(req)
    @test payload["request"]["generationConfig"]["temperature"] == 0.5
    @test payload["request"]["systemInstruction"][:parts][1][:text] == "You are a cat."
end

@testset "BatchRequest sends tools and safety settings" begin
    lookup(; city) = city
    config = GenerateContentConfig(;
        tools=Any[google_search_tool(), lookup],
        tool_config=ToolConfig(;
            function_calling_config=FunctionCallingConfig(; mode="ANY")
        ),
        safety_settings=[SafetySetting("HARM_CATEGORY_HARASSMENT", "BLOCK_NONE")],
    )
    conversation = [Dict(:role => "user", :parts => [Dict(:text => "hi")])]
    request = GoogleGenAI._batch_request_payload(BatchRequest(conversation; config), 3)
    @test request["metadata"]["key"] == "request-3"
    body = request["request"]
    @test body["contents"] === conversation
    @test haskey(body["tools"][1], "googleSearch")
    @test body["tools"][2]["functionDeclarations"][1]["name"] == "lookup"
    @test body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    @test only(body["safetySettings"]).threshold == "BLOCK_NONE"
end

@testset "Inline requests default keys" begin
    requests = [BatchRequest("a"), BatchRequest("b"; key="custom")]
    body = GoogleGenAI._build_batch_request_body(; requests)
    payloads = body["batch"]["input_config"]["requests"]["requests"]
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
    @test body["batch"]["input_config"] == Dict("file_name" => "files/123456")
    @test !haskey(body["batch"], "display_name")
end

@testset "Batch endpoints" begin
    requests = []
    handler = function (req)
        push!(
            requests,
            (target=req.target, body=isempty(req.body) ? nothing : JSON3.read(req.body)),
        )
        return json_response(
            Dict("name" => "batches/1", "operations" => [Dict("name" => "batches/1")])
        )
    end
    with_server(handler) do provider
        create_batch_job(
            provider, "gemini-x"; requests=[BatchRequest("hi")], display_name="gen"
        )
        create_embeddings_batch_job(
            provider,
            "embed-x";
            contents=["a", "b"],
            task_type="CLUSTERING",
            display_name="emb",
        )
        create_embeddings_batch_job(provider, "embed-x"; input_file="files/in")
        @test [j.name for j in list_batch_jobs(provider; page_size=10)] == ["batches/1"]
        @test_throws ArgumentError create_embeddings_batch_job(provider, "embed-x")
    end
    gen, emb, emb_file, listed = requests
    @test gen.target == "/v1beta/models/gemini-x:batchGenerateContent"
    @test gen.body.batch.display_name == "gen"
    @test emb.target == "/v1beta/models/embed-x:asyncBatchEmbedContent"
    inline = emb.body.batch.input_config.requests.requests
    @test [r.request.content.parts[1].text for r in inline] == ["a", "b"]
    @test all(
        r -> r.request.taskType == "CLUSTERING" && r.request.model == "models/embed-x",
        inline,
    )
    @test [r.metadata.key for r in inline] == ["request-1", "request-2"]
    @test emb_file.body.batch.input_config.file_name == "files/in"
    @test listed.target == "/v1beta/batches?pageSize=10"
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
            "inlinedResponses": {
                "inlinedResponses": [
                    {
                        "metadata": {"key": "request-1"},
                        "response": {
                            "candidates": [{"content": {"parts": [
                                {"text": "Hello"}, {"text": " world"}
                            ]}}]
                        }
                    },
                    {
                        "metadata": {"key": "request-2"},
                        "error": {"code": 400, "message": "bad request"}
                    }
                ]
            }
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
