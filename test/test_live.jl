using GoogleGenAI
using Base64: base64encode
using JSON3
using Test

@testset "Live setup message" begin
    config = GenerateContentConfig(;
        response_modalities=["AUDIO"],
        system_instruction="Be brief.",
        temperature=0.5,
        speech_config=Dict(
            :voiceConfig => Dict(:prebuiltVoiceConfig => Dict(:voiceName => "Kore"))
        ),
    )
    setup = GoogleGenAI._build_live_setup("gemini-2.5-flash", config)

    @test setup["model"] == "models/gemini-2.5-flash"
    @test setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    @test setup["generationConfig"]["temperature"] == 0.5
    @test haskey(setup["generationConfig"]["speechConfig"], :voiceConfig)
    @test setup["systemInstruction"][:parts][1][:text] == "Be brief."
    @test !haskey(setup, "tools")

    minimal = GoogleGenAI._build_live_setup("m", GenerateContentConfig())
    @test minimal == Dict("model" => "models/m")

    thinking = GoogleGenAI._build_live_setup(
        "m",
        GenerateContentConfig(;
            thinking_config=ThinkingConfig(; thinking_budget=0), service_tier="flex"
        ),
    )
    @test thinking["generationConfig"]["thinkingConfig"] == Dict("thinkingBudget" => 0)
    @test !haskey(thinking, "serviceTier")
end

@testset "Live setup options" begin
    setup = GoogleGenAI._build_live_setup(
        "m",
        GenerateContentConfig();
        input_audio_transcription=true,
        output_audio_transcription=true,
        session_resumption=true,
        context_window_compression=true,
        realtime_input_config=Dict(:automaticActivityDetection => Dict(:disabled => true)),
    )
    @test setup["inputAudioTranscription"] == Dict()
    @test setup["outputAudioTranscription"] == Dict()
    @test setup["sessionResumption"] == Dict()
    @test setup["contextWindowCompression"] == Dict("slidingWindow" => Dict())
    @test setup["realtimeInputConfig"][:automaticActivityDetection][:disabled]

    resumed = GoogleGenAI._build_live_setup(
        "m",
        GenerateContentConfig();
        session_resumption="handle-1",
        context_window_compression=Dict(:triggerTokens => 1000),
    )
    @test resumed["sessionResumption"] == Dict("handle" => "handle-1")
    @test resumed["contextWindowCompression"] == Dict(:triggerTokens => 1000)
    @test !haskey(resumed, "inputAudioTranscription")
    @test_throws MethodError GoogleGenAI._build_live_setup(
        "m", GenerateContentConfig(); unknown=1
    )
end

@testset "Live event parsing (transcription and resumption)" begin
    session = GoogleGenAI.LiveSession(
        GoogleProvider(api_key="k"), "m", GenerateContentConfig(), Dict{String,Function}()
    )
    event = GoogleGenAI._parse_live_event(
        session,
        """{"serverContent": {"inputTranscription": {"text": "hello"}, "outputTranscription": {"text": "PONG"}}}""",
    )
    @test event[:input_transcription] == "hello"
    @test event[:output_transcription] == "PONG"

    GoogleGenAI._parse_live_event(
        session, """{"sessionResumptionUpdate": {"newHandle": "h1", "resumable": true}}"""
    )
    @test session.resumption_handle == "h1"
    GoogleGenAI._parse_live_event(
        session, """{"sessionResumptionUpdate": {"resumable": false}}"""
    )
    @test session.resumption_handle == "h1"
end

@testset "Live and REST build the same tools" begin
    weather = FunctionDeclaration(;
        name="weather",
        description="Get current weather",
        parameters=FunctionParameter(;
            type="object", properties=Dict{String,Any}(), required=["city"]
        ),
    )
    lookup(; city) = city
    config = GenerateContentConfig(;
        tools=Any[Dict(:googleSearch => Dict()), lookup],
        function_declarations=[weather],
        tool_config=ToolConfig(;
            function_calling_config=FunctionCallingConfig(; mode="ANY")
        ),
    )
    setup = GoogleGenAI._build_live_setup("m", config)
    tools = setup["tools"]
    @test length(tools) == 2
    @test haskey(tools[1], "googleSearch")
    @test [d["name"] for d in tools[2]["functionDeclarations"]] == ["lookup", "weather"]
    @test setup["toolConfig"] == Dict("functionCallingConfig" => Dict("mode" => "ANY"))

    body = GoogleGenAI._build_request_body(Dict{Symbol,Any}[], config)
    @test body["tools"] == tools
    @test body["toolConfig"] == setup["toolConfig"]
end

@testset "Live websocket URL" begin
    provider = GoogleProvider(api_key="k")
    url = GoogleGenAI._live_websocket_url(provider)
    @test startswith(url, "wss://generativelanguage.googleapis.com/ws/")
    @test endswith(url, "BidiGenerateContent")  # key goes in a header
end

@testset "Live event parsing (auto tool call)" begin
    session_calls = String[]
    functions = Dict{String,Function}(
        "get_weather" => function (; city)
            push!(session_calls, city)
            return Dict{String,Any}("temp_c" => 25)
        end
    )
    session = GoogleGenAI.LiveSession(
        GoogleProvider(api_key="k"), "m", GenerateContentConfig(), functions
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
        GoogleProvider(api_key="k"), "m", GenerateContentConfig(), Dict{String,Function}()
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
        GoogleProvider(api_key="k"), "m", GenerateContentConfig(), Dict{String,Function}()
    )

    send_realtime_input(
        session;
        audio=UInt8[0x01],
        audio_mime_type="audio/pcm;rate=16000",
        video=UInt8[0x02],
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

    send_tool_response(
        session, Dict(:id => "7", :name => "f", :response => Dict(:ok => true))
    )
    msg = JSON3.read(take!(session.out_channel))
    @test msg.toolResponse.functionResponses[1].id == "7"

    @test_throws ArgumentError send_realtime_input(session)
end
