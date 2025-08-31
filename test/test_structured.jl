@testset "Structured Generation" begin
    model = "gemini-2.5-flash-lite"
    config = GenerateContentConfig(; http_options=http_options)

    schema = Dict(
        :type => "ARRAY",
        :items => Dict(
            :type => "OBJECT",
            :properties => Dict(
                :recipe_name => Dict(:type => "STRING"),
                :ingredients =>
                    Dict(:type => "ARRAY", :items => Dict(:type => "STRING")),
            ),
            :propertyOrdering => ["recipe_name", "ingredients"],
        ),
    )

    config = GenerateContentConfig(;
        response_mime_type="application/json", response_schema=schema
    )

    prompt = "List a few popular cookie recipes with exact amounts of each ingredient."
    response = generate_content(secret_key, model, prompt; config=config)
    json_string = response.text
    recipes = JSON3.read(json_string)
    @test length(recipes) > 0
    @test haskey(recipes[1], "recipe_name")
    @test haskey(recipes[1], "ingredients")
end
