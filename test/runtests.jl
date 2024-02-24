using Aqua
using GoogleGenAI
using Test

const secret_key = ENV["GOOGLE_API_KEY"]

@testset "GoogleGenAI.jl" begin
    # Generate text from text
    response = generate_content(secret_key, "gemini-pro", "Hello"; max_output_tokens=50)
    @test typeof(response) == GoogleGenAI.GoogleTextResponse

    # Generate text from text+image
    response = generate_content(
        secret_key,
        "gemini-pro-vision",
        "What is this picture?",
        "example.jpg";
        max_output_tokens=50,
    )
    @test typeof(response) == GoogleGenAI.GoogleTextResponse

    # Multi-turn conversation
    conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
    response = generate_content(
        secret_key, "gemini-pro", conversation; max_output_tokens=50
    )

    n_tokens = count_tokens(secret_key, "gemini-pro", "Hello")
    @test n_tokens == 1

    embeddings = embed_content(secret_key, "embedding-001", "Hello")
    @test typeof(embeddings) == GoogleGenAI.GoogleEmbeddingResponse
    @test size(embeddings.values) == (768,)

    models = list_models(secret_key)
    @test length(models) > 0
    @test haskey(models[1], :name)
end

Aqua.test_all(GoogleGenAI)
