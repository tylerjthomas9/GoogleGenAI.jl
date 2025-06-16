@testset "Image Generation" begin
    config = GenerateContentConfig(; response_modalities=["Text", "Image"])
    prompt = (
        "Hi, can you create a 3d rendered image of a pig " *
        "with wings and a top hat flying over a happy " *
        "futuristic scifi city with lots of greenery?"
    )

    response = generate_content(
        secret_key, "gemini-2.0-flash-exp-image-generation", prompt; config
    )
    @test !isempty(response.images)

    image_path = "input/example.jpg"
    model = "gemini-2.0-flash-exp-image-generation"
    prompt = "Make all of the circles green"
    response = generate_content(secret_key, model, prompt; image_path, config)
    @test !isempty(response.images)
end
