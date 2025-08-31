
@testset "Code Generation" begin
    model = "gemini-2.5-flash"
    tools = [Dict(:code_execution => Dict())]
    config = GenerateContentConfig(; http_options, tools)

    prompt = "Write a function to calculate the factorial of a number."
    response = generate_content(secret_key, model, prompt; config=config)
    @test response.response_status == 200
    @test response.text isa String
    @test occursin("```python", response.text)
end

@testset "Basic Function Calling" begin
    # Define a function declaration for weather information
    weather_function = FunctionDeclaration(
        "get_weather",
        "Get current weather information for a location",
        Dict{String,Any}(
            "type" => "object",
            "properties" => Dict{String,Any}(
                "location" => Dict{String,Any}(
                    "type" => "string",
                    "description" => "City or location to get weather for",
                ),
                "unit" => Dict{String,Any}(
                    "type" => "string",
                    "description" => "Temperature unit (celsius or fahrenheit)",
                    "enum" => ["celsius", "fahrenheit"],
                ),
            ),
            "required" => ["location"],
        ),
    )

    # Test the function declaration itself
    @test weather_function.name == "get_weather"
    @test weather_function.parameters.type == "object"
    @test haskey(weather_function.parameters.properties, "location")

    # Configure to force function calling
    fc_config = FunctionCallingConfig(; mode="ANY")
    tool_config = ToolConfig(; function_calling_config=fc_config)
    config = GenerateContentConfig(;
        http_options=http_options,
        function_declarations=[weather_function],
        tool_config=tool_config,
        temperature=0.2,
    )
    function get_weather(; location::String, unit::String="celsius")
        return Dict("temperature" => 25, "condition" => "sunny", "unit" => unit)
    end

    # Create a test conversation
    conversation = [
        Dict(
            :role => "user", :parts => [Dict(:text => "What's the weather like in Paris?")]
        ),
    ]

    model = "gemini-2.5-flash"
    response = generate_content(secret_key, model, conversation; config=config)
    @test response.response_status == 200
    @test response.function_calls !== nothing

    # Execute function
    fc = response.function_calls[1]
    result = get_weather(; location=get(fc.args, "location", "unknown"))

    # Add to conversation with model's function call
    model_msg = Dict(
        :role => "model",
        :parts => [Dict(:functionCall => Dict(:name => fc.name, :args => fc.args))],
    )

    # Add function response
    func_msg = Dict(
        :role => "function",
        :parts => [Dict(:functionResponse => Dict(:name => fc.name, :response => result))],
    )

    # Create new conversation
    new_convo = [conversation[1], model_msg, func_msg]
    final_config = GenerateContentConfig(; http_options=http_options, temperature=0.7)
    final_response = generate_content(secret_key, model, new_convo; config=final_config)

    # Verify final response
    @test final_response.response_status == 200
    @test occursin("25", final_response.text)
end

@testset "Function Tools Types" begin
    weather_function = FunctionDeclaration(
        "get_weather",
        "Get weather information",
        Dict{String,Any}(
            "type" => "object",
            "properties" => Dict{String,Any}(
                "location" => Dict{String,Any}(
                    "type" => "string", "description" => "Location name"
                ),
            ),
            "required" => ["location"],
        ),
    )
    api_decl = GoogleGenAI.to_api_function_declaration(weather_function)
    function_tool = Dict{Symbol,Any}(:functionDeclarations => [api_decl])
    code_tool = Dict{Symbol,Any}(:codeExecution => Dict{String,Any}())
    search_tool = Dict{Symbol,Any}(:googleSearch => Dict{String,Any}())

    # Test tool type detection
    is_native, tool_type = is_native_tool(function_tool)
    @test is_native == true
    @test tool_type == FUNCTION_CALLING

    is_native, tool_type = is_native_tool(code_tool)
    @test is_native == true
    @test tool_type == CODE_EXECUTION

    is_native, tool_type = is_native_tool(search_tool)
    @test is_native == true
    @test tool_type == GOOGLE_SEARCH
end

@testset "Function Calling Config" begin
    # Test function calling modes
    auto_config = FunctionCallingConfig(; mode="AUTO")
    @test auto_config.mode == "AUTO"
    @test auto_config.allowed_function_names === nothing

    any_config = FunctionCallingConfig(; mode="ANY", allowed_function_names=["get_weather"])
    @test any_config.mode == "ANY"
    @test any_config.allowed_function_names == ["get_weather"]

    none_config = FunctionCallingConfig(; mode="NONE")
    @test none_config.mode == "NONE"

    # Test invalid mode
    @test_throws ArgumentError FunctionCallingConfig(mode="INVALID")

    # Test tool config creation
    tool_config = ToolConfig(; function_calling_config=auto_config)
    @test tool_config.function_calling_config === auto_config
end

@testset "Utility Functions" begin
    # Test string_to_symbol_keys
    str_dict = Dict("name" => "value", "number" => 42)
    sym_dict = GoogleGenAI.string_to_symbol_keys(str_dict)

    @test haskey(sym_dict, :name)
    @test haskey(sym_dict, :number)
    @test sym_dict[:name] == "value"
    @test sym_dict[:number] == 42

    # Test function conversation building
    user_query = "What's the weather?"
    function_name = "get_weather"
    function_args = Dict("location" => "Paris")
    function_result = Dict("temperature" => 25, "condition" => "sunny")

    convo = GoogleGenAI.build_function_conversation(
        user_query, function_name, function_args, function_result
    )

    @test length(convo) == 3
    @test convo[1][:role] == "user"
    @test convo[2][:role] == "model"
    @test convo[3][:role] == "function"
    @test haskey(convo[2][:parts][1], :functionCall)
    @test haskey(convo[3][:parts][1], :functionResponse)

    # Test adding function result to conversation
    base_convo = [Dict(:role => "user", :parts => [Dict(:text => "Hello")])]
    updated_convo = add_function_result_to_conversation(
        base_convo, function_name, function_result
    )

    @test length(updated_convo) == 2
    @test updated_convo[2][:role] == "function"
    @test haskey(updated_convo[2][:parts][1], :functionResponse)
end

@testset "Parallel Function Calling" begin
    # Define parallel functions
    light_function = FunctionDeclaration(
        "control_lights",
        "Control the smart lighting system",
        Dict{String,Any}(
            "type" => "object",
            "properties" => Dict{String,Any}(
                "brightness" => Dict{String,Any}(
                    "type" => "number",
                    "description" => "Light brightness from 0.0 to 1.0",
                ),
                "color" => Dict{String,Any}(
                    "type" => "string",
                    "description" => "Light color (red, blue, etc)",
                ),
            ),
            "required" => ["brightness"],
        ),
    )

    music_function = FunctionDeclaration(
        "play_music",
        "Play music on speakers",
        Dict{String,Any}(
            "type" => "object",
            "properties" => Dict{String,Any}(
                "genre" => Dict{String,Any}(
                    "type" => "string", "description" => "Music genre to play"
                ),
                "volume" => Dict{String,Any}(
                    "type" => "number", "description" => "Volume from 0.0 to 1.0"
                ),
            ),
            "required" => ["genre", "volume"],
        ),
    )

    # Function implementations
    function control_lights(; brightness::Float64, color::String="white")
        return Dict(
            "brightness" => brightness, "color" => color, "status" => "Lights adjusted"
        )
    end

    function play_music(; genre::String, volume::Float64)
        return Dict("genre" => genre, "volume" => volume, "status" => "Music playing")
    end

    # Mock function calls
    function_calls = [
        FunctionCall("control_lights", Dict("brightness" => 0.5, "color" => "blue")),
        FunctionCall("play_music", Dict("genre" => "jazz", "volume" => 0.7)),
    ]

    # Test execution
    functions = Dict("control_lights" => control_lights, "play_music" => play_music)

    results = execute_parallel_function_calls(function_calls, functions)

    # Verify results
    @test length(results) == 2
    @test haskey(results, "control_lights")
    @test haskey(results, "play_music")
    @test haskey(results["control_lights"], "status")
    @test haskey(results["play_music"], "status")

    # Test error handling
    invalid_function_calls = [
        FunctionCall("nonexistent_function", Dict("param" => "value"))
    ]

    error_results = execute_parallel_function_calls(invalid_function_calls, functions)
    @test haskey(error_results, "nonexistent_function")
    @test haskey(error_results["nonexistent_function"], "error")
end
