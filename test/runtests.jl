using Aqua
using GoogleGenAI
using Test

if haskey(ENV, "GOOGLE_API_KEY")
    const secret_key = ENV["GOOGLE_API_KEY"]

    @testset "GoogleGenAI.jl" begin
        # Generate text from text
        response = generate_content(secret_key, "gemini-pro", "Hello"; max_output_tokens=50)

        # Generate text from text+image
        response = generate_content(
            secret_key,
            "gemini-pro-vision",
            "What is this picture?",
            "example.jpg";
            max_output_tokens=50,
        )

        # Multi-turn conversation
        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(
            secret_key, "gemini-pro", conversation; max_output_tokens=50
        )

        n_tokens = count_tokens(secret_key, "gemini-pro", "Hello")
        @test n_tokens == 1

        embeddings = embed_content(secret_key, "embedding-001", "Hello")
        @test size(embeddings.values) == (768,)

        embeddings = embed_content(secret_key, "embedding-001", ["Hello", "world"])
        @test size(embeddings.values[1]) == (768,)
        @test size(embeddings.values[2]) == (768,)

        models = list_models(secret_key)
        @test length(models) > 0
        @test haskey(models[1], :name)
    end
else
    @info "Skipping GoogleGenAI.jl tests because GOOGLE_API_KEY is not set"
end

Aqua.test_all(GoogleGenAI)
