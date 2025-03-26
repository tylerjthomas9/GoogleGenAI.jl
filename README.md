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

1. Create a [secret API key in Google AI Studio](https://aistudio.google.com).
2. Set the `GOOGLE_API_KEY` environment variable.

### Generate Content

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-2.0-flash"
prompt = "Hello"
response = generate_content(secret_key, model, prompt)
println(response.text)
```

Gemini API config:
```julia
config = GenerateContentConfig(; max_output_tokens=50)
response = generate_content(secret_key, model, prompt; config)
println(response.text)
```

Single image input:
```julia
prompt = "What is this image?"
image_path = "test/input/example.jpg"
response = generate_content(secret_key, model, prompt; image_path)
println(response.text)
```

### Multi-turn conversations

```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
config = GenerateContentConfig(; max_output_tokens=50)
model = "gemini-2.0-flash"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "When was Julia 1.0 released?")])
]

response = generate_content(provider, model, conversation; config)
push!(conversation, Dict(:role => "model", :parts => [Dict(:text => response.text)]))
println("Model: ", response.text) 

push!(conversation, Dict(:role => "user", :parts => [Dict(:text => "Who created the language?")]))
response = generate_content(provider, model, conversation; config)
println("Model: ", response.text)
```

### Streaming Content Generation

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-2.0-flash"
prompt = "Why is the sky blue?"

# Get a channel that yields partial results
stream = generate_content_stream(secret_key, model, prompt)

# Process the stream as results arrive
ix = 0
for chunk in stream
    print(chunk.text)
end
```

For multi-turn conversations with streaming:

```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
model = "gemini-2.0-flash"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "Write a short poem about Julia programming language")])
]

# First message
println("Generating first response...")
stream = generate_content_stream(provider, model, conversation)
last_response = ""

for chunk in stream
    println("Response: ", chunk.text)
end
```

### Generate/Edit Images

Generate image using Gemini:
```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
config = GenerateContentConfig(
    response_modalities=["Text", "Image"]
)

prompt = ("Hi, can you create a 3d rendered image of a pig "*
            "with wings and a top hat flying over a happy "*
            "futuristic scifi city with lots of greenery?")

response = generate_content(
    secret_key,
    "gemini-2.0-flash-exp-image-generation",
    prompt;
    config
);

if !isempty(response.images)
    open("gemini-native-image.png", "w") do io
        write(io, response.images[1].data)
    end
end
```

Edit image with Gemini:
```julia
image_path = "gemini-native-image.png"

model = "gemini-2.0-flash-exp-image-generation"
prompt = "Make the pig a llama"
response = generate_content(
    secret_key,
    model,
    prompt;
    image_path,
    config
);

if !isempty(response.images)
    open("gemini-native-image-edited.png", "w") do io
        write(io, response.images[1].data)
    end
end
```

### Count Tokens
```julia
using GoogleGenAI
model = "gemini-2.0-flash"
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
embeddings = embed_content(ENV["GOOGLE_API_KEY"], "text-embedding-004", "Hello")
println(size(embeddings.values))
```
outputs
```julia
(768,)
```

```julia
using GoogleGenAI
embeddings = embed_content(ENV["GOOGLE_API_KEY"], "text-embedding-004", ["Hello", "world"])
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
gemini-1.0-pro-vision-latest
gemini-pro-vision
gemini-1.5-pro-latest
gemini-1.5-pro-001
gemini-1.5-pro-002
gemini-1.5-pro
gemini-1.5-flash-latest
gemini-1.5-flash-001
gemini-1.5-flash-001-tuning
gemini-1.5-flash
gemini-1.5-flash-002
gemini-1.5-flash-8b
gemini-1.5-flash-8b-001
gemini-1.5-flash-8b-latest
gemini-1.5-flash-8b-exp-0827
gemini-1.5-flash-8b-exp-0924
gemini-2.5-pro-exp-03-25
gemini-2.0-flash-exp
gemini-2.0-flash
gemini-2.0-flash-001
gemini-2.0-flash-exp-image-generation
gemini-2.0-flash-lite-001
gemini-2.0-flash-lite
gemini-2.0-flash-lite-preview-02-05
gemini-2.0-flash-lite-preview
gemini-2.0-pro-exp
gemini-2.0-pro-exp-02-05
gemini-exp-1206
gemini-2.0-flash-thinking-exp-01-21
gemini-2.0-flash-thinking-exp
gemini-2.0-flash-thinking-exp-1219
learnlm-1.5-pro-experimental
gemma-3-27b-it
```

### Safety Settings

More information about the safety settings can be found [here](https://ai.google.dev/gemini-api/docs/safety-settings).

```julia
using GoogleGenAI
secret_key = ENV["GOOGLE_API_KEY"]
safety_settings = [
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="HARM_BLOCK_THRESHOLD_UNSPECIFIED"),
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
    SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_LOW_AND_ABOVE"),
]
model = "gemini-1.5-flash-latest"
prompt = "Hello"
config = GenerateContentConfig(; safety_settings)
response = generate_content(secret_key, model, prompt; config)
```


### Content Caching

List models that support content caching:

```julia
using GoogleGenAI
models = list_models(ENV["GOOGLE_API_KEY"])
for m in models
    if "createCachedContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end
```
```julia
gemini-1.5-pro-001
gemini-1.5-pro-002
gemini-1.5-flash-001
gemini-1.5-flash-002
gemini-1.5-flash-8b
gemini-1.5-flash-8b-001
gemini-1.5-flash-8b-latest
```

Cache content to reuse it across multiple requests:

```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
model = "gemini-1.5-flash-002"

# Create cached content (at least 32,786 tokens are required for caching)
text = read("test/input/example.txt", String) ^ 7
cache_result = create_cached_content(
    provider,
    model,
    text,
    ttl="90s", # Cache for 90 seconds
)

# Now generate content that references the cached content.
prompt = "Please summarize this document"
config = GenerateContentConfig(; cached_content=cache_result.name)
response = generate_content(
    provider,
    model,
    prompt;
    config
)
println(response.text)


# list all cached content
list_result = list_cached_content(provider)
# get details of a specific cache
get_result = get_cached_content(provider, cache_result.name)
# update the TTL of a specific cache
update_result = update_cached_content(provider, cache_result.name, "90s") 
# delete a specific cache
delete_cached_content(provider, cache_result.name)
```

### Files

Files are only supported in Gemini Developer API.


```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
file_path = "test/input/example.jpg"

# upload file
upload_result = upload_file(
    provider, file_path; display_name="Test File",
)

# generate content with file
model = "gemini-2.0-flash-lite"
prompt = "What is this image?"
contents = [prompt, upload_result]
response = generate_content(
    provider,
    model,
    contents;
)
println(response.text)

# Get file metadata
get_result = get_file(provider, upload_result[:name])

# List files
list_result = list_files(provider)

# Delete file
delete_file(provider, upload_result[:name])
```

## Structured Generation


Json 
```julia
using GoogleGenAI
using JSON3

# API key and model
api_key = ENV["GOOGLE_API_KEY"]
model   = "gemini-2.0-flash"

# Define a JSON schema for an Array of Objects
# Each object has "recipe_name" (a String) and "ingredients" (an Array of Strings).
schema = Dict(
    :type => "ARRAY",
    :items => Dict(
        :type => "OBJECT",
        :properties => Dict(
            :recipe_name => Dict(:type => "STRING"),
            :ingredients => Dict(
                :type  => "ARRAY",
                :items => Dict(:type => "STRING")
            )
        ),
        :propertyOrdering => ["recipe_name", "ingredients"]
    )
)

config = GenerateContentConfig(
    response_mime_type = "application/json",
    response_schema    = schema,
)

prompt = "List a few popular cookie recipes with exact amounts of each ingredient."
response = generate_content(api_key, model, prompt; config=config)
json_string = response.text
recipes = JSON3.read(json_string)
println(recipes)
```
outputs
```julia
JSON3.Object[{
   "recipe_name": "Chocolate Chip Cookies",
   "ingredients": [
                    "1 cup (2 sticks) unsalted butter, softened",
                    "3/4 cup granulated sugar",
                    "3/4 cup packed brown sugar",
                    "1 teaspoon vanilla extract",
                    "2 large eggs",
                    "2 1/4 cups all-purpose flour",
                    "1 teaspoon baking soda",
                    "1 teaspoon salt",
                    "2 cups chocolate chips"
                  ]
}, {
   "recipe_name": "Peanut Butter Cookies",
   "ingredients": [
                    "1 cup (2 sticks) unsalted butter, softened",
                    "1 cup creamy peanut butter",
                    "1 cup granulated sugar",
                    "1 cup packed brown sugar",
                    "2 large eggs",
                    "1 teaspoon vanilla extract",
                    "2 1/2 cups all-purpose flour",
                    "1 teaspoon baking soda",
                    "1/2 teaspoon salt"
                  ]
}, {
   "recipe_name": "Sugar Cookies",
   "ingredients": [
                    "1 1/2 cups (3 sticks) unsalted butter, softened",
                    "2 cups granulated sugar",
                    "4 large eggs",
                    "1 teaspoon vanilla extract",
                    "5 cups all-purpose flour",
                    "2 teaspoons baking powder",
                    "1 teaspoon salt"
                  ]
}]
```

# Code Generation

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-2.0-flash"

tools = [Dict(:code_execution => Dict())]
config = GenerateContentConfig(; tools)

prompt = "Write a function to calculate the factorial of a number."
response = generate_content(secret_key, model, prompt; config=config)
println(response.text)
```
