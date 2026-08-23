using GoogleGenAI
using Base64: base64encode
using JSON3
using Test

@testset "Live setup message" begin
    config = GenerateContentConfig(;
        response_modalities=["AUDIO"],
        system_instruction="Be brief.",
        temperature=0.5,
        speech_config=Dict(:voiceConfig => Dict(:prebuiltVoiceConfig => Dict(:voiceName => "Kore"))),
    )
    setup = GoogleGenAI._build_live_setup("gemini-2.5-flash", config)

    @test setup["model"] == "models/gemini-2.5-flash"
    @test setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    @test setup["generationConfig"]["temperature"] == 0.5
    @test haskey(setup["generationConfig"]["speechConfig"], "voiceConfig")
    @test setup["systemInstruction"]["parts"][1]["text"] == "Be brief."
    @test !haskey(setup, "tools")

    # Defaults are omitted
    minimal = GoogleGenAI._build_live_setup("m", GenerateContentConfig())
    @test minimal["model"] == "models/m"
    @test !haskey(minimal, "generationConfig")
    @test !haskey(minimal, "systemInstruction")
end

@testset "Live tools assembly" begin
    weather = FunctionDeclaration(;
        name="weather",
        description="Get current weather",
        parameters=FunctionParameter(; type="object", properties=Dict{String,Any}(), required=["city"]),
    )
    config = GenerateContentConfig(; tools=Any[Dict(:googleSearch => Dict())], function_declarations=[weather])
    tools = GoogleGenAI._build_live_tools(config)
    @test length(tools) == 2
    @test haskey(tools[1], "googleSearch")
    @test tools[2]["functionDeclarations"][1]["name"] == "weather"
end

@testset "Live websocket URL" begin
    provider = GoogleProvider(api_key="k")
    url = GoogleGenAI._live_websocket_url(provider)
    @test startswith(url, "wss://generativelanguage.googleapis.com/ws/")
    @test occursin("BidiGenerateContent?key=k", url)
end

@testset "Live event parsing (auto tool call)" begin
    session_calls = String[]
    functions = Dict{String,Function}("get_weather" => function (; city)
        push!(session_calls, city)
        return Dict{String,Any}("temp_c" => 25)
    end)
    session = GoogleGenAI.LiveSession(
        GoogleProvider(api_key="k"),
        "m",
        GenerateContentConfig(),
        functions,
        Channel{String}(Inf),
        Channel{Dict{Symbol,Any}}(64),
        Base.RefValue{Any}(nothing),
        true,
        true,
    )
    frame = """{"toolCall": {"functionCalls": [{"id": "1", "name": "get_weather", "args": {"city": "Oslo"}}]}}"""
    event = GoogleGenAI._parse_live_event(session, frame)
    @test haskey(event, :toolCall)
    @test session_calls == ["Oslo"]

    response_frame = take!(session.out_channel)
    msg = JSON3.read(response_frame)
    @test msg.toolResponse.functionResponses[1].id == "1"
    @test msg.toolResponse.functionResponses[1].response["temp_c"] == 25
end

@testset "Live event parsing (audio extraction)" begin
    session = GoogleGenAI.LiveSession(
        GoogleProvider(api_key="k"),
        "m",
        GenerateContentConfig(),
        Dict{String,Function}(),
        Channel{String}(Inf),
        Channel{Dict{Symbol,Any}}(64),
        Base.RefValue{Any}(nothing),
        true,
        true,
    )
    audio_b64 = base64encode(UInt8[1, 2, 3])
    frame = """{"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "$audio_b64"}}]}, "turnComplete": false}}"""
    event = GoogleGenAI._parse_live_event(session, frame)
    mime, bytes = event[:audio]
    @test mime == "audio/pcm;rate=24000"
    @test bytes == UInt8[1, 2, 3]
end

@testset "Live send helpers serialize frames" begin
    session = GoogleGenAI.LiveSession(
        GoogleProvider(api_key="k"),
        "m",
        GenerateContentConfig(),
        Dict{String,Function}(),
        Channel{String}(Inf),
        Channel{Dict{Symbol,Any}}(64),
        Base.RefValue{Any}(nothing),
        true,
        true,
    )

    send_realtime_input(
        session; audio=UInt8[0x01], audio_mime_type="audio/pcm;rate=16000", video=UInt8[0x02]
    )
    msg = JSON3.read(take!(session.out_channel))
    chunks = msg.realtimeInput.mediaChunks
    @test length(chunks) == 2
    @test chunks[1].mimeType == "audio/pcm;rate=16000"
    @test chunks[2].mimeType == "image/jpeg"

    send_realtime_input(session; activity_start=true)
    @test haskey(JSON3.read(take!(session.out_channel)).realtimeInput, :activityStart)

    send_client_content(session; turns="Hi", turn_complete=true)
    msg = JSON3.read(take!(session.out_channel))
    @test msg.clientContent.turns[1].parts[1].text == "Hi"
    @test msg.clientContent.turnComplete === true

    send_tool_response(session, Dict(:id => "7", :name => "f", :response => Dict(:ok => true)))
    msg = JSON3.read(take!(session.out_channel))
    @test msg.toolResponse.functionResponses[1].id == "7"

    @test_throws ArgumentError send_realtime_input(session)
end
