 [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
 [![CI](https://github.com/tylerjthomas9/GoogleGenAI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tylerjthomas9/GoogleGenAI.jl/actions/workflows/CI.yml)
 [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
 [![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://tylerjthomas9.github.io/GoogleGenAI.jl)


# GoogleGenAI.jl

## Overview

A Julia wrapper to the Google generative AI API. For API functionality, see [reference documentation](https://ai.google.dev/tutorials/rest_quickstart).

## Installation

From source:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/tylerjthomas9/GoogleGenAI.jl/")
```

```julia
julia> ]  # enters the pkg interface
Pkg> add https://github.com/tylerjthomas9/GoogleGenAI.jl/
```

## Quick Start

Create a [secret API key in Google AI Studio](https://makersuite.google.com/)


### Generate Content

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-1.5-flash-latest"
prompt = "Hello"
response = generate_content(secret_key, model, prompt)
println(response.text)
```
outputs
```julia
"Hello! 👋  How can I help you today? 😊"
```

```julia
api_kwargs = (max_output_tokens=50,)
response = generate_content(secret_key, model, prompt; api_kwargs)
println(response.text)
```
outputs
```julia
"Hello! 👋  How can I help you today? 😊"
```

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-pro-vision"
prompt = "What is this image?"
image_path = "test/example.jpg"
response = generate_content(secret_key, model, prompt, image_path)
println(response.text)
```
outputs
```julia
"The logo for the Julia programming language."
```

### Multi-turn conversations

```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
api_kwargs = (max_output_tokens=50,)
model = "gemini-1.5-flash-latest"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "When was Julia 1.0 released?")])
]

response = generate_content(provider, model, conversation)
push!(conversation, Dict(:role => "model", :parts => [Dict(:text => response.text)]))
println("Model: ", response.text) 

push!(conversation, Dict(:role => "user", :parts => [Dict(:text => "Who created the language?")]))
response = generate_content(provider, model, conversation; api_kwargs)
println("Model: ", response.text)
```
outputs
```julia
"Model: Julia 1.0 was released on **August 8, 2018**."

"Model: Julia was created by a team of developers at MIT, led by **Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and Alan Edelman**."
```

### Count Tokens
```julia
using GoogleGenAI
model = "gemini-1.5-flash-latest"
n_tokens = count_tokens(ENV["GOOGLE_API_KEY"], model, "The Julia programming language")
println(n_tokens)
```
outputs
```julia
4
```

### Create Embeddings

```julia
using GoogleGenAI
embeddings = embed_content(ENV["GOOGLE_API_KEY"], "embedding-001", "Hello")
println(size(embeddings.values))
```
outputs
```julia
(768,)
```

```julia
using GoogleGenAI
embeddings = embed_content(ENV["GOOGLE_API_KEY"], "embedding-001", ["Hello", "world"])
println(embeddings.response_status)
println(size(embeddings.values[1]))
println(size(embeddings.values[2]))
```
outputs
```julia
200
(768,)
(768,)
```

### List Models

```julia
using GoogleGenAI
models = list_models(ENV["GOOGLE_API_KEY"])
for m in models
    if "generateContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end
```
outputs
```julia
gemini-1.0-pro
gemini-1.0-pro-001
gemini-1.0-pro-latest
gemini-1.0-pro-vision-latest
gemini-1.5-flash-latest
gemini-1.5-pro-latest
gemini-pro
gemini-pro-vision
```

### Safety Settings

More information about the safety settings can be found [here](https://ai.google.dev/docs/safety_setting_gemini).

```julia
using GoogleGenAI
secret_key = ENV["GOOGLE_API_KEY"]
safety_settings = [
    Dict("category" => "HARM_CATEGORY_HATE_SPEECH", "threshold" => "HARM_BLOCK_THRESHOLD_UNSPECIFIED"),
    Dict("category" => "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold" => "BLOCK_ONLY_HIGH"),
    Dict("category" => "HARM_CATEGORY_HARASSMENT", "threshold" => "BLOCK_MEDIUM_AND_ABOVE"),
    Dict("category" => "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold" => "BLOCK_LOW_AND_ABOVE")
]
model = "gemini-1.5-flash-latest"
prompt = "Hello"
api_kwargs = (safety_settings=safety_settings,)
response = generate_content(secret_key, model, prompt; api_kwargs)
```
