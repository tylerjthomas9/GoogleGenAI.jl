@testset "Response Parsing: Safety-Blocked Response" begin
    response = GoogleGenAI._parse_response(FakeResponse(load_fixture("blocked.json")))

    # A SAFETY finish reason still carries text
    @test response.finish_reason == "SAFETY"
    @test !isempty(response.text)
    @test response.response_status == 200
    # Ratings are reported per-candidate; _parse_response surfaces them
    # from candidates[1], falling back to the top-level field.
    @test length(response.safety_ratings) == 2
end

@testset "Embedding Response Parsing" begin
    # embed_content reads "embedding"."values" straight off the response body;
    # verify the same access pattern works against the saved payload.
    body = JSON3.read(load_fixture("embedding.json"))
    values = get(get(body, "embedding", Dict()), "values", Vector{Float64}())

    @test length(values) == 5
    @test values[1] ≈ 0.1
end

@testset "Image Part Parsing" begin
    # 1x1 red PNG, base64-encoded — the smallest valid inline image payload.
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    body = """
    {
      "candidates": [{
        "content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "$png_b64"}}]},
        "finishReason": "STOP"
      }]
    }
    """
    response = GoogleGenAI._parse_response(FakeResponse(body))

    @test isempty(response.text)
    @test length(response.images) == 1
    img = only(response.images)
    @test img.mime_type == "image/png"
    @test img.data[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]  # PNG magic bytes
end
