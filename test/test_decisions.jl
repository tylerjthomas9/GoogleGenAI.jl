@testset "Decisions: question serialization" begin
    @test GoogleGenAI._question_dict(NoulQuestion("Damaged?")) ==
        Dict("type" => "noul", "instructions" => "Damaged?")
    @test GoogleGenAI._question_dict(
        NoulQuestion("Damaged?"; true_description="Broken", false_description="Intact")
    )["criteria"] == Dict("true_description" => "Broken", "false_description" => "Intact")
    @test GoogleGenAI._question_dict(
        NoulQuestion("Damaged?"; true_description="Broken")
    )["criteria"] == Dict("true_description" => "Broken")

    choice = GoogleGenAI._question_dict(ChoiceQuestion("Topic", ["shipping", "billing"]))
    @test choice["type"] == "choice"
    @test choice["criteria"] == Dict("shipping" => "", "billing" => "")
    described = ChoiceQuestion("Topic", Dict(:shipping => "Late or lost", :billing => ""))
    @test described.criteria == Dict("shipping" => "Late or lost", "billing" => "")

    score = GoogleGenAI._question_dict(ScoreQuestion("Severity", ["low", "mid", "high"]))
    @test score == Dict(
        "type" => "score",
        "instructions" => "Severity",
        "criteria" => ["low", "mid", "high"],
    )

    @test_throws ArgumentError ChoiceQuestion("Topic", ["only"])
    @test_throws ArgumentError ChoiceQuestion("Topic", ["same", "same"])
    @test_throws ArgumentError ScoreQuestion("Severity", ["only"])
end

@testset "Decisions: request body" begin
    raw = Dict("type" => "noul", "instructions" => "raw")
    body = GoogleGenAI._build_decision_body(
        "gemini-decision-1", "Some text", (a=NoulQuestion("A?"), b=raw)
    )
    @test body["model"] == "gemini-decision-1"
    @test body["input"] == "Some text"
    @test body["questions"]["a"] == Dict("type" => "noul", "instructions" => "A?")
    @test body["questions"]["b"] === raw

    dict_body = GoogleGenAI._build_decision_body(
        "m", "x", Dict("topic-v2" => NoulQuestion("A?"))
    )
    @test haskey(dict_body["questions"], "topic-v2")

    @test_throws ArgumentError GoogleGenAI._build_decision_body("m", "x", NamedTuple())
end

@testset "Decisions: input blocks" begin
    image_path = joinpath(@__DIR__, "input", "example.jpg")
    file = JSON3.read(
        """{"uri": "https://example.com/files/abc", "mimeType": "audio/mp3"}"""
    )
    input = GoogleGenAI._decision_input([
        "Look:",
        (path=image_path, resolution="low"),
        (data=UInt8[1, 2, 3], mime_type="image/png"),
        (data="AQID", mime_type="application/pdf"),
        (uri="https://example.com/v.mp4", mime_type="video/mp4"),
        file,
    ])

    @test input[1] == Dict("type" => "text", "text" => "Look:")
    @test input[2]["type"] == "image"
    @test input[2]["mime_type"] == "image/jpeg"
    @test input[2]["resolution"] == "low"
    @test input[2]["data"] == base64encode(read(image_path))
    @test input[3] == Dict("type" => "image", "mime_type" => "image/png", "data" => "AQID")
    @test input[4]["type"] == "document"
    @test input[5] == Dict(
        "type" => "video", "mime_type" => "video/mp4", "uri" => "https://example.com/v.mp4"
    )
    @test input[6] == Dict(
        "type" => "audio",
        "mime_type" => "audio/mp3",
        "uri" => "https://example.com/files/abc",
    )

    @test_throws ArgumentError GoogleGenAI._input_block((uri="https://example.com/x",))
    @test_throws ArgumentError GoogleGenAI._input_block((mime_type="image/png",))
    @test_throws ArgumentError GoogleGenAI._input_block((
        data="AQID", mime_type="text/plain"
    ))
end

@testset "Decisions: response parsing" begin
    r = GoogleGenAI._parse_decision_response(FakeResponse(load_fixture("decision.json")))

    @test r.response_status == 200
    @test r.prompt_feedback === nothing
    @test r.usage[:total_input_tokens] == 255

    noul = r.answers["wants_cancel"]
    @test noul isa NoulAnswer
    @test noul.noul ≈ 0.9746116115316582
    @test noul.confidence ≈ noul.noul

    choice = r.answers["topic"]
    @test choice isa ChoiceAnswer
    @test choice.choice == "support_quality"
    @test sum(values(choice.probabilities)) ≈ 1
    @test choice.probabilities["billing"] ≈ 0.018688503300978997

    score = r.answers["sentiment"]
    @test score isa ScoreAnswer
    @test score.legend ==
        ["very negative", "negative", "neutral", "positive", "very positive"]
    @test score.probabilities[1] ≈ score.confidence
    @test length(score.probabilities) == 5
    @test score.score ≈ 0.03637631237455299

    @test r.answers["future"] isa JSON3.Object
end
