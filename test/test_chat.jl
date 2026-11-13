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
    @test conversation[2][:parts][1][:functionResponse][:response] ==
        Dict("temperature" => 25)
end

@testset "Function call execution" begin
    weather(; location) = location == "Tokyo" ? Dict("temperature" => 25) : error("unknown")
    functions = Dict{String,Function}("get_weather" => weather)
    call(name, args) = GoogleGenAI._call_function(functions, name, args)

    @test call("get_weather", Dict("location" => "Tokyo")) == Dict("temperature" => 25)
    @test call("get_weather", JSON3.read("""{"location": "Tokyo"}""")) ==
        Dict("temperature" => 25)
    @test haskey(call("get_weather", Dict("location" => "Paris")), "error")
    @test call("missing", Dict()) == Dict("error" => "Function missing not found")

    returning(x) = Dict{String,Function}("f" => (; kw...) -> x)
    obj = JSON3.read("""{"a": 1}""")
    @test GoogleGenAI._call_function(returning(obj), "f", Dict()) === obj
    @test GoogleGenAI._call_function(returning(nothing), "f", Dict()) ==
        Dict("status" => "completed")
    @test GoogleGenAI._call_function(returning(42), "f", Dict()) == Dict("value" => 42)

    results = execute_parallel_function_calls(
        [FunctionCall("f", Dict{String,Any}())], returning(obj)
    )
    @test results["f"] === obj
end
