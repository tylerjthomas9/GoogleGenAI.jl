 [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
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
model = "gemini-pro"
prompt = "Hello"
response = generate_content(secret_key, model, prompt)
println(response.text)
```
outputs
```julia
"Hello there! How may I assist you today? Feel free to ask me any questions you may have or give me a command. I'm here to help! 😊"
```

```julia
response = generate_content(secret_key, model, prompt; max_output_tokens=10)
println(response.text)
```
outputs
```julia
"Hello! How can I assist you today?"
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
# Define the provider with your API key (placeholder here)
provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
model_name = "gemini-pro"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "When was Julia 1.0 released?")])
]

response = generate_content(provider, model_name, conversation)
push!(conversation, Dict(:role => "model", :parts => [Dict(:text => response.text)]))
println("Model: ", response.text) 

push!(conversation, Dict(:role => "user", :parts => [Dict(:text => "Who created the language?")]))
response = generate_content(provider, model_name, conversation, max_output_tokens=100)
println("Model: ", response.text)
```
outputs
```julia
"Model: August 8, 2018"

"Model: Jeff Bezanson, Alan Edelman, Viral B. Shah, Stefan Karpinski, and Keno Fischer

Julia Computing, Inc. is the company that provides commercial support for Julia."
```

### Count Tokens
```julia
using GoogleGenAI
n_tokens = count_tokens(ENV["GOOGLE_API_KEY"], "gemini-pro", "Hello")
println(n_tokens)
```
outputs
```julia
1
```

### Create Embeddings

```julia
using GoogleGenAI
embeddings = create_embeddings(ENV["GOOGLE_API_KEY"], "gemini-pro", "Hello")
println(size(embeddings.values))
```
outputs
```julia
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
model = "gemini-pro"
prompt = "Hello"
response = generate_content(secret_key, model, prompt; safety_settings=safety_settings)
```
