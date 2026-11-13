function with_sse_server(f, streams)
    bodies = []
    server = HTTP.serve!("127.0.0.1", 18753; listenany=true, stream=true) do http
        push!(bodies, JSON3.read(read(http)))
        payload = join("data: $(JSON3.write(e))\r\n\r\n" for e in streams[length(bodies)])
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for i in 1:37:lastindex(payload)
            write(http, payload[i:min(i + 36, end)])
            sleep(0.005)
        end
    end
    try
        port = HTTP.Servers.port(server)
        f(GoogleProvider(; api_key="test", base_url="http://127.0.0.1:$port"), bodies)
    finally
        close(server)
    end
end

function model_event(parts...; finish_reason=nothing)
    content = Dict("role" => "model", "parts" => collect(parts))
    return Dict(
        "candidates" => [Dict("content" => content, "finishReason" => finish_reason)]
    )
end

text_part(t) = Dict("text" => t)

@testset "SSE line splitting keeps partial lines" begin
    buf = Vector{UInt8}(codeunits("data: one\r\n\r\ndata: tw"))
    @test strip.(GoogleGenAI._take_lines!(buf)) == ["data: one", ""]
    @test String(copy(buf)) == "data: tw"

    append!(buf, codeunits("o\r\n"))
    @test strip.(GoogleGenAI._take_lines!(buf)) == ["data: two"]
    @test isempty(buf)

    @test isempty(GoogleGenAI._take_lines!(buf))
end

@testset "Streaming keeps repeated and empty chunks" begin
    texts = ["Hello", " the", " the", "\n\n", "\n\n", "", "end"]
    final_event = Dict(
        "candidates" =>
            [Dict("content" => Dict("role" => "model"), "finishReason" => "STOP")],
        "usageMetadata" => Dict("totalTokenCount" => 7),
    )
    events = [[model_event(text_part(t)) for t in texts]; final_event]

    with_sse_server([events]) do provider, _
        chunks = collect(generate_content_stream(provider, "test-model", "hi"))

        @test !any(c -> haskey(c, :error), chunks)
        @test [c.text for c in chunks] == [texts; ""]
        @test chunks[end].full_text == join(texts)
        @test chunks[end].finish_reason == "STOP"
        @test chunks[end].usage_metadata.totalTokenCount == 7
    end
end

@testset "Chat: streaming records the whole reply" begin
    call = Dict(
        "functionCall" =>
            Dict("name" => "get_weather", "args" => Dict("location" => "Tokyo")),
        "thoughtSignature" => "sig",
    )
    streams = [
        [
            model_event(text_part("Hello")),
            model_event(text_part(" the")),
            model_event(text_part(" the")),
            model_event(text_part(" end"); finish_reason="STOP"),
        ],
        [
            model_event(text_part("Checking. ")),
            model_event(call),
            model_event(text_part(""); finish_reason="STOP"),
        ],
        [
            model_event(text_part("It's 25")),
            model_event(text_part(" degrees."); finish_reason="STOP"),
        ],
    ]
    functions = Dict{String,Function}(
        "get_weather" => (; location) -> Dict("temperature" => 25)
    )

    with_sse_server(streams) do provider, bodies
        chat = create_chat(provider, "test-model"; functions)
        @test !any(c -> haskey(c, :error), collect(send_message_stream(chat, "hi")))
        @test chat.history[2][:parts] == [Dict(:text => "Hello the the end")]

        chunks = collect(send_message_stream(chat, "Weather in Tokyo?"))
        @test !any(c -> haskey(c, :error), chunks)
        @test bodies[2].contents[2].parts[1].text == "Hello the the end"

        call_turn = bodies[3].contents[4]
        @test call_turn.role == "model"
        @test length(call_turn.parts) == 2
        @test call_turn.parts[1].text == "Checking. "
        @test call_turn.parts[2].functionCall.name == "get_weather"
        @test call_turn.parts[2].thoughtSignature == "sig"
        @test bodies[3].contents[5].parts[1].functionResponse.response.temperature == 25

        @test length(chat.history) == 4
        @test chat.history[4][:parts] == [Dict(:text => "It's 25 degrees.")]
    end
end
