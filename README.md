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
model = "gemini-2.0-flash"
prompt = "What is this image?"
image_path = "test/example.jpg"
response = generate_content(secret_key, model, prompt; image_path)
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
model = "gemini-2.0-flash"
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

### Streaming Content Generation

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-2.0-flash"
prompt = "Write a short story about a magic backpack"

# Get a channel that yields partial results
stream = generate_content_stream(secret_key, model, prompt)

# Process the stream as results arrive
for chunk in stream
    println(chunk.text)
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
gemini-2.0-flash-exp
gemini-2.0-flash
gemini-2.0-flash-001
gemini-2.0-flash-lite-001
gemini-2.0-flash-lite
gemini-2.0-pro-exp
gemini-2.0-pro-exp-02-05
gemini-exp-1206
gemini-2.0-flash-thinking-exp-01-21
gemini-2.0-flash-thinking-exp
gemini-2.0-flash-thinking-exp-1219
learnlm-1.5-pro-experimental
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
text = read("test/example.txt", String) ^ 7
cache_result = create_cached_content(
    provider,
    model,
    text,
    ttl="360s", # Cache for 60 seconds
    # system_instruction="You are Julia's Number 1 fan",
)

# Now generate content that references the cached content.
prompt = "Please summarize this document"
config = GenerateContentConfig(; cached_content=cache_name)
response = generate_content(
    provider,
    model,
    prompt;
    config
)
println(response.text)
```

### Files

Files are only supported in Gemini Developer API.


```julia
using GoogleGenAI

provider = GoogleProvider(api_key=ENV["GOOGLE_API_KEY"])
file_path = "test/example.jpg"

# upload file
upload_result = upload_file(
    provider, file_path; display_name="Test JPEG", mime_type="image/jpeg"
)

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
model = "gemini-2.0-flash-lite"

tools = [Dict(:code_execution => Dict())]
config = GenerateContentConfig(; tools)

prompt = "Write a function to calculate the factorial of a number."
response = generate_content(secret_key, model, prompt; config=config)
println(response.text)
```

```bash
Okay, I will write a function to calculate the factorial of a number.

Here's the Python code:

```python
def factorial(n):
    """
    This function calculates the factorial of a non-negative integer.

    Args:
    n: A non-negative integer.

    Returns:
    The factorial of n (n!), or 1 if n is 0.  Returns None for negative input.
    """
    if n < 0:
        return None # Factorial is not defined for negative numbers
    elif n == 0:
        return 1  # Base case: factorial of 0 is 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
```
```
