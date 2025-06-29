# GoogleGenAI.jl Docs

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

### API Key Setup

To use the Gemini API, you need an API key. You can create a key for free with a few clicks in [Google AI Studio](https://aistudio.google.com). For more details, see the [API key documentation](https://ai.google.dev/gemini-api/docs/api-key).

There are several ways to provide your API key:

### Option 1: Environment Variables (Recommended)
Set your API key as an environment variable `GOOGLE_API_KEY` or `GEMINI_API_KEY` (if both are set, `GOOGLE_API_KEY` takes precedence):

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Option 2: Explicit API Key as String
Pass the API key directly as a string parameter:

```julia
api_key = "your-api-key-here"
response = generate_content(api_key, model, prompt)
```

### Option 3: GoogleProvider with API Key
Create a `GoogleProvider` instance with your API key:

```julia
provider = GoogleProvider(; api_key="your-api-key-here")
response = generate_content(provider, model, prompt)
```

### Generate Content

```julia
using GoogleGenAI

model = "gemini-2.5-flash"
prompt = "Hello"
response = generate_content(model, prompt)
println(response.text)
```

Gemini API config:
```julia
config = GenerateContentConfig(; max_output_tokens=50)
response = generate_content(model, prompt; config)
println(response.text)
```

Single image input:
```julia
prompt = "What is this image?"
image_path = "test/input/example.jpg"
response = generate_content(model, prompt; image_path)
println(response.text)
```

### Multi-turn conversations

```julia
using GoogleGenAI

config = GenerateContentConfig(; max_output_tokens=50)
model = "gemini-2.5-flash"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "When was Julia 1.0 released?")])
]

response = generate_content( model, conversation; config)
push!(conversation, Dict(:role => "model", :parts => [Dict(:text => response.text)]))
println("Model: ", response.text) 

push!(conversation, Dict(:role => "user", :parts => [Dict(:text => "Who created the language?")]))
response = generate_content(model, conversation; config)
println("Model: ", response.text)
```

### Streaming Content Generation

```julia
using GoogleGenAI

model = "gemini-2.5-flash"
prompt = "Why is the sky blue?"

stream = generate_content_stream(model, prompt)

for chunk in stream
    print(chunk.text)
end
```

For multi-turn conversations with streaming:

```julia
using GoogleGenAI

provider = GoogleProvider()
model = "gemini-2.5-flash"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "Write a short poem about Julia programming language")])
]

println("Generating first response...")
stream = generate_content_stream(provider, model, conversation)

for chunk in stream
    println("Response: ", chunk.text)
end
```

### Generate/Edit Images

Generate image using Gemini:
```julia
using GoogleGenAI

config = GenerateContentConfig(
    response_modalities=["Text", "Image"]
)

prompt = ("Hi, can you create a 3d rendered image of a pig "*
            "with wings and a top hat flying over a happy "*
            "futuristic scifi city with lots of greenery?")

response = generate_content(
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

model = "gemini-2.5-flash"
n_tokens = count_tokens(model, "The Julia programming language")
println(n_tokens)
```
outputs
```julia
4
```

### Create Embeddings

```julia
using GoogleGenAI

embeddings = embed_content("text-embedding-004", "Hello")
println(size(embeddings.values))
```
outputs
```julia
(768,)
```

```julia
using GoogleGenAI

embeddings = embed_content("text-embedding-004", ["Hello", "world"])
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

models = list_models()
for m in models
    if "generateContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end
```
outputs
```julia
gemini-2.5-pro-preview-03-25
gemini-2.5-flash-preview-04-17
gemini-2.5-flash
gemini-2.5-flash
gemini-2.5-flash-preview-04-17-thinking
gemini-2.5-flash-lite-preview-06-17
gemini-2.5-pro-preview-05-06
gemini-2.5-pro-preview-06-05
gemini-2.5-pro
gemini-2.0-flash-exp
gemini-2.0-flash
gemini-2.0-flash-001
gemini-2.0-flash-exp-image-generation
gemini-2.0-flash-lite-001
gemini-2.0-flash-lite
gemini-2.0-flash-preview-image-generation
gemini-2.0-flash-lite-preview-02-05
gemini-2.0-flash-lite-preview
gemini-2.0-pro-exp
gemini-2.0-pro-exp-02-05
gemini-exp-1206
gemini-2.0-flash-thinking-exp-01-21
gemini-2.0-flash-thinking-exp
gemini-2.0-flash-thinking-exp-1219
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
learnlm-2.0-flash-experimental
gemma-3-1b-it
gemma-3-4b-it
gemma-3-12b-it
gemma-3-27b-it
gemma-3n-e4b-it
gemma-3n-e2b-it
```

### Safety Settings

More information about the safety settings can be found [here](https://ai.google.dev/gemini-api/docs/safety-settings).

```julia
using GoogleGenAI

safety_settings = [
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="HARM_BLOCK_THRESHOLD_UNSPECIFIED"),
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"),
    SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
]
model = "gemini-2.5-flash"
prompt = "Hello"
config = GenerateContentConfig(; safety_settings)
response = generate_content(model, prompt; config)
```

### Thinking 

The Gemini 2.5 series models use an internal "thinking process" during response generation. This process contributes to their improved reasoning capabilities and helps them use multi-step planning to solve complex tasks. This thinking can be limited by setting the `thinking_budget`. 

```julia
using GoogleGenAI

thinking_config = ThinkingConfig(; thinking_budget=100, include_thoughts=true)
config = GenerateContentConfig(;
    thinking_config
)
model = "gemini-2.5-flash"
response = generate_content(model, "Hello"; config)
```


### Content Caching

List models that support content caching:

```julia
using GoogleGenAI

models = list_models()
for m in models
    if "createCachedContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end
```
```julia
gemini-2.5-pro-preview-03-25
gemini-2.5-flash-preview-04-17
gemini-2.5-flash-preview-05-20
gemini-2.5-flash
gemini-2.5-flash-preview-04-17-thinking
gemini-2.5-flash-lite-preview-06-17
gemini-2.5-pro-preview-05-06
gemini-2.5-pro-preview-06-05
gemini-2.5-pro
gemini-2.0-flash
gemini-2.0-flash-001
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
```

Cache content to reuse it across multiple requests:

```julia
using GoogleGenAI

# API key is automatically loaded from environment variable
provider = GoogleProvider()
model = "gemini-2.5-flash"

# Create cached content
text = read("test/input/example.txt", String)
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

# upload file
file_path = "test/input/example.jpg"
upload_result = upload_file(
    file_path; display_name="Test File",
)

# generate content with file
model = "gemini-2.5-flash"
prompt = "What is this image?"
contents = [prompt, upload_result]
response = generate_content(
    model,
    contents;
)
println(response.text)

# Get file metadata
get_result = get_file(upload_result[:name])

# List files
list_result = list_files()

# Delete file
delete_file(upload_result[:name])
```

## Structured Generation

Json 
```julia
using GoogleGenAI
using JSON3

model = "gemini-2.5-flash"

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
response = generate_content(model, prompt; config=config)
json_string = response.text
recipes = JSON3.read(json_string)
println(recipes)
```

outputs
```julia
JSON3.Object[{
   "recipe_name": "Classic Chocolate Chip Cookies",
   "ingredients": [
                    "1 cup (2 sticks) unsalted butter, softened",
                    "3/4 cup granulated sugar",
                    "3/4 cup packed light brown sugar",
                    "2 large eggs",
                    "1 teaspoon vanilla extract",
                    "2 1/4 cups all-purpose flour",
                    "1 teaspoon baking soda",
                    "1/2 teaspoon salt",
                    "1 cup (6 oz) semi-sweet chocolate chips"
                  ]
}, {
   "recipe_name": "Soft Oatmeal Raisin Cookies",
   "ingredients": [
                    "1 cup (2 sticks) unsalted butter, softened",
                    "1 cup packed light brown sugar",
                    "1/2 cup granulated sugar",
                    "2 large eggs",
                    "1 teaspoon vanilla extract",
                    "1 1/2 cups all-purpose flour",
                    "1 teaspoon baking soda",
                    "1/2 teaspoon ground cinnamon",
                    "1/2 teaspoon salt",
                    "3 cups rolled oats",
                    "1 cup raisins"
                  ]
}, {
   "recipe_name": "Simple Peanut Butter Cookies",
   "ingredients": [
                    "1 cup (2 sticks) unsalted butter, softened",
                    "1 cup creamy peanut butter",
                    "1 cup packed light brown sugar",
                    "1 cup granulated sugar",
                    "2 large eggs",
                    "1 teaspoon vanilla extract",
                    "2 1/2 cups all-purpose flour",
                    "1 teaspoon baking soda",
                    "1/2 teaspoon salt"
                  ]
}]
```

# Code Generation

```julia
using GoogleGenAI

tools = [Dict(:code_execution => Dict())]
config = GenerateContentConfig(; tools)

model = "gemini-2.5-flash"
prompt = "Write a function to calculate the factorial of a number."
response = generate_content(model, prompt; config=config)
println(response.text)
```

# Function Calling

## Manually declare and invoke a function for function calling

```julia
using GoogleGenAI
using JSON3

# Step 1: Create the initial user message
user_message = Dict(
    :role => "user", 
    :parts => [Dict(:text => "What's the weather like in Paris?")]
)

# Step 2: Define your function declaration
weather_function = FunctionDeclaration(
    "get_weather", 
    "Get current weather information for a location",
    Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "location" => Dict{String, Any}(
                "type" => "string",
                "description" => "City or location to get weather for"
            ),
            "unit" => Dict{String, Any}(
                "type" => "string",
                "description" => "Temperature unit (celsius or fahrenheit)",
                "enum" => ["celsius", "fahrenheit"]
            )
        ),
        "required" => ["location"]
    )
)

# Step 3: Configure the model to force function calling
fc_config = FunctionCallingConfig(mode="ANY")
tool_config = ToolConfig(function_calling_config=fc_config)
config = GenerateContentConfig(
    function_declarations=[weather_function],
    tool_config=tool_config,
    temperature=0.2
)

# Step 4: Get the initial response from the model, which should be a function call
response = generate_content(
    "gemini-2.5-flash",
    [user_message];
    config=config
)

# Step 5: Extract the function call details and construct the model's turn
function_name = response.function_calls[1].name
args = response.function_calls[1].args
model_message = Dict(
    :role => "model",
    :parts => [
        Dict(
            :functionCall => Dict(
                :name => function_name,
                :args => args
            )
        )
    ]
)
println("\nModel message: ", JSON3.write(model_message))

# Step 6: Execute the function (simulated) and create the function's response message
# In a real app, you would call your actual get_weather function here.
weather_result = Dict(
    "temperature" => 18,
    "condition" => "Sunny",
    "humidity" => 65
)

function_message = Dict(
    :role => "function",
    :parts => [
        Dict(
            :functionResponse => Dict(
                :name => function_name,
                :response => weather_result
            )
        )
    ]
)
println("\nFunction message: ", JSON3.write(function_message))

# Step 7: Assemble the full conversation history
conversation_history = [
    user_message,
    model_message,
    function_message
]

# Step 8: Get the final, natural language response from the model
final_response = generate_content(
    "gemini-2.5-flash",
    conversation_history
)

println("\nFinal response: $(final_response.text)")
```
