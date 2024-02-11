module GoogleGenAI

using JSON3
using HTTP

Base.@kwdef struct GoogleProvider
    api_key::String = ""
    base_url::String = "https://generativelanguage.googleapis.com/v1beta"
end

struct GoogleResponse
    candidates::Vector{Dict{Symbol, Any}}
    safety_ratings::Dict{Pair{Symbol, String}, Pair{Symbol, String}}
    text::String
end

struct BlockedPromptException <: Exception end

function _extract_text(response::JSON3.Object)
    all_texts = String[] 
    for candidate in response.candidates
        candidate_text = join([part.text for part in candidate.content.parts], "")
        push!(all_texts, candidate_text)
    end
    return all_texts
end

function generate_content(provider::GoogleProvider, model_name::String, input::String)
    url = "$(provider.base_url)/models/$model_name:generateContent?key=$(provider.api_key)"
    body = Dict("contents" => [Dict("parts" => [Dict("text" => input)])])

    response = HTTP.post(url, headers = Dict("Content-Type" => "application/json"), body = JSON3.write(body))
    
    if response.status >= 200 && response.status < 300
        parsed_response = JSON3.read(response.body)
        all_texts = _extract_text(parsed_response)
        concatenated_texts = join(all_texts, "")
        candidates = [Dict(i) for i in parsed_response[:candidates]]
        safety_rating = Dict(parsed_response.promptFeedback.safetyRatings)
        return GoogleResponse(candidates, safety_rating, concatenated_texts)
    else
        error("Request failed with status $(response.status): $(String(response.body))")
    end
end
generate_content(api_key::String, model_name::String, input::String) = generate_content(GoogleProvider(; api_key), model_name, input)

export GoogleProvider, GoogleResponse, generate_content

end # module GoogleGenerativeAI
