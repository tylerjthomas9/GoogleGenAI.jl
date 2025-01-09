using Aqua
using GoogleGenAI
using Test

if haskey(ENV, "GOOGLE_API_KEY")
    const secret_key = ENV["GOOGLE_API_KEY"]

    @testset "GoogleGenAI.jl" begin
        model = "gemini-2.0-flash-exp"
        embedding_model = "text-embedding-004"
        api_kwargs = (max_output_tokens=50,)
        http_kwargs = (retries=2,)
        # Generate text from text
        response = generate_content(secret_key, model, "Hello"; api_kwargs, http_kwargs)

        # Generate text from text+image
        response = generate_content(
            secret_key,
            model,
            "What is this picture?",
            "example.jpg";
            api_kwargs,
            http_kwargs,
        )

        # Multi-turn conversation
        conversation = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
        response = generate_content(
            secret_key, model, conversation; api_kwargs, http_kwargs
        )

        n_tokens = count_tokens(secret_key, model, "Hello")
        @test n_tokens == 1

        embeddings = embed_content(secret_key, embedding_model, "Hello")
        @test size(embeddings.values) == (768,)

        embeddings = embed_content(secret_key, embedding_model, ["Hello", "world"])
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
