@testset "Chat: Construction & History" begin
    chat = create_chat("fake-key", "gemini-3.7-flash")
    @test chat isa Chat
    @test chat.model == "gemini-3.7-flash"
    @test isempty(chat.history)
    @test isempty(chat.functions)

    history = Dict{Symbol,Any}[
        Dict(:role => "user", :parts => [Dict(:text => "hi")]),
        Dict(:role => "model", :parts => [Dict(:text => "hello")]),
    ]
    chat = create_chat("fake-key", "gemini-3.7-flash"; history=history)
    @test length(get_history(chat)) == 2
    @test get_history(chat) !== chat.history

    clear_history!(chat)
    @test isempty(get_history(chat))
end

@testset "Chat: Curated History" begin
    chat = create_chat("fake-key", "gemini-3.7-flash")
    chat.history = Dict{Symbol,Any}[
        Dict(:role => "user", :parts => [Dict(:text => "What's the weather?")]),
        Dict(
            :role => "model",
            :parts =>
                [Dict(:functionCall => Dict(:name => "get_weather", :args => Dict()))],
        ),
        Dict(
            :role => "user",
            :parts => [
                Dict(
                    :functionResponse =>
                        Dict(:name => "get_weather", :response => Dict("temp" => 25)),
                ),
            ],
        ),
        Dict(:role => "model", :parts => [Dict(:text => "It's 25 degrees.")]),
    ]

    curated = get_history(chat; curated=true)
    @test length(curated) == 2
    @test curated[1][:role] == "user"
    @test curated[2][:parts][1] == Dict(:text => "It's 25 degrees.")
end

@testset "Chat: Automatic Function Calling" begin
    functions = Dict{String,Function}(
        "get_weather" =>
            (; location::String) ->
                (location == "Tokyo" ? Dict("temperature" => 25) : error("unknown")),
    )
    fc = FunctionCall(; name="get_weather", args=Dict{String,Any}("location" => "Tokyo"))

    @test GoogleGenAI._execute_function(functions, fc) == Dict("temperature" => 25)
    @test GoogleGenAI._execute_function(
        functions, FunctionCall(; name="missing", args=Dict{String,Any}())
    ) == Dict("error" => "Function missing not found")

    conversation = Dict{Symbol,Any}[]
    GoogleGenAI._record_function_exchange!(
        conversation,
        functions,
        [Dict(:functionCall => Dict(:name => fc.name, :args => fc.args))],
        [fc],
    )
    @test conversation[1][:role] == "model"
    @test conversation[1][:parts][1][:functionCall][:name] == "get_weather"
    @test conversation[2][:role] == "user"
    @test conversation[2][:parts][1][:functionResponse][:name] == "get_weather"

    # Result serialization rules
    @test GoogleGenAI._serialize_function_result(Dict("a" => 1)) == Dict("a" => 1)
    @test GoogleGenAI._serialize_function_result(nothing) == Dict("status" => "completed")
    @test GoogleGenAI._serialize_function_result(42) == Dict("value" => 42)
end
