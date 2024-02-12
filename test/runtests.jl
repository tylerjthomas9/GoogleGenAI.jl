using Aqua
using GoogleGenAI
using Test

const secret_key = ENV["GOOGLE_API_KEY"]

@testset "GoogleGenAI.jl" begin
    response = generate_content(secret_key, "gemini-pro", "Hello")
    @test typeof(response) == GoogleGenAI.GoogleTextResponse

    n_tokens = count_tokens(secret_key, "gemini-pro", "Hello")
    @test n_tokens == 1

    embeddings = embed_content(secret_key, "embedding-001", "Hello")
    @test typeof(embeddings) == GoogleGenAI.GoogleEmbeddingResponse
    @test size(embeddings.values) == (768,)
end

Aqua.test_all(GoogleGenAI)
