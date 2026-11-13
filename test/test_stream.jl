@testset "SSE line splitting keeps partial lines" begin
    buf = Vector{UInt8}(codeunits("data: one\r\n\r\ndata: tw"))
    @test strip.(GoogleGenAI._take_lines!(buf)) == ["data: one", ""]
    @test String(copy(buf)) == "data: tw"

    append!(buf, codeunits("o\r\n"))
    @test strip.(GoogleGenAI._take_lines!(buf)) == ["data: two"]
    @test isempty(buf)

    @test isempty(GoogleGenAI._take_lines!(buf))
end

# A local server stands in for streamGenerateContent. It writes the SSE
# payload in small pieces so events are split across network reads.
@testset "Streaming keeps repeated and empty chunks" begin
    texts = ["Hello", " the", " the", "\n\n", "\n\n", "", "end"]
    text_event(t) = Dict(
        "candidates" => [
            Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => t)])),
        ],
    )
    final_event = Dict(
        "candidates" =>
            [Dict("content" => Dict("role" => "model"), "finishReason" => "STOP")],
        "usageMetadata" => Dict("totalTokenCount" => 7),
    )
    events = [text_event.(texts); final_event]
    payload = join("data: $(JSON3.write(e))\r\n\r\n" for e in events)
    pieces = [payload[i:min(i + 36, end)] for i in 1:37:lastindex(payload)]

    server = HTTP.serve!("127.0.0.1", 18753; listenany=true, stream=true) do http
        read(http)  # HTTP.jl logs an EOFError if the request body is left unread
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for piece in pieces
            write(http, piece)
            sleep(0.005)
        end
    end
    try
        provider = GoogleProvider(;
            api_key="test", base_url="http://127.0.0.1:$(HTTP.Servers.port(server))"
        )
        chunks = collect(generate_content_stream(provider, "test-model", "hi"))

        @test !any(c -> haskey(c, :error), chunks)
        @test [c.text for c in chunks] == [texts; ""]
        @test chunks[end].full_text == join(texts)
        @test chunks[end].finish_reason == "STOP"
        @test chunks[end].usage_metadata.totalTokenCount == 7
    finally
        close(server)
    end
end
