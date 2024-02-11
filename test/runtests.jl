using Aqua
using GoogleGenerativeAI
using Test

@testset "GoogleGenerativeAI.jl" begin
    model = GenerativeModel("gemini-pro")
    response = generate_content(model, "Hello, how are you?")
end

Aqua.test_all(GoogleGenerativeAI)
