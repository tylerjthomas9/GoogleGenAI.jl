using Aqua
using GoogleGenAI
using Test

const secret_key = ENV["GOOGLE_API_KEY"]

@testset "GoogleGenAI.jl" begin
    response = generate_content(secret_key, "gemini-pro", "Hello, how are you?")
end

Aqua.test_all(GoogleGenAI)
