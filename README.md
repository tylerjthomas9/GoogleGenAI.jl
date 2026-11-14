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

### Retries, Timeouts and Errors

Failed requests throw a `GeminiAPIError` with the HTTP `code`, the API's `status` (e.g. `"RESOURCE_EXHAUSTED"`) and `message`. Requests are not retried unless the provider has `retry_options`; `http_options` sets default `HTTP.request` options such as timeouts for every call.

```julia
provider = GoogleProvider(;
    retry_options=RetryOptions(),  # up to 5 attempts on 408, 429, 500, 502, 503, 504
    http_options=(readtimeout=120,),
)

try
    generate_content(provider, model, prompt)
catch e
    e isa GeminiAPIError && e.code == 429 && @warn "Rate limited" e.message e.retry_delay
end
```

### Generate Content

```julia
using GoogleGenAI

model = "gemini-3.7-flash"
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

Single image input (legacy):
```julia
prompt = "What is this image?"
image_path = "test/input/example.jpg"
response = generate_content(model, prompt; image_path)
println(response.text)
```

Multi image input:
```julia
prompt = "What is this image?"
image_path = "test/input/example.jpg"
images = [(path=image_path,), (path=image_path, mime_type="image/png")]
response = generate_content(model, prompt; images)
println(response.text)
```

Or mix text, media (`path`, `data` or `uri`, with `mime_type`) and uploaded files in one turn:
```julia
contents = ["Compare these:", (path="test/input/example.jpg",), (data=pdf_bytes, mime_type="application/pdf")]
response = generate_content(model, contents)
```

The system instruction can be a string, a vector of strings and parts, or a content dict:
```julia
config = GenerateContentConfig(; system_instruction=["You are a pirate.", "Answer in rhyme."])
```


### Multi-turn conversations

```julia
using GoogleGenAI

config = GenerateContentConfig(; max_output_tokens=50)
model = "gemini-3.7-flash"
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

model = "gemini-3.7-flash"
prompt = "Why is the sky blue?"

stream = generate_content_stream(model, prompt)

for chunk in stream
    haskey(chunk, :error) && throw(chunk.error)
    print(chunk.text)
end
```

For multi-turn conversations with streaming:

```julia
using GoogleGenAI

provider = GoogleProvider()
model = "gemini-3.7-flash"
conversation = [
    Dict(:role => "user", :parts => [Dict(:text => "Write a short poem about Julia programming language")])
]

println("Generating first response...")
stream = generate_content_stream(provider, model, conversation)

for chunk in stream
    haskey(chunk, :error) && throw(chunk.error)
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
    "gemini-3.1-flash-image",
    prompt;
    config
);

if !isempty(response.images)
    open("gemini-native-image.png", "w") do io
        write(io, response.images[1].data)
    end
end
```

Set the aspect ratio and size with `image_config`:
```julia
config = GenerateContentConfig(;
    response_modalities=["Image"], image_config=Dict(:aspectRatio => "16:9", :imageSize => "2K")
)
```

Edit image with Gemini:
```julia
image_path = "gemini-native-image.png"

model = "gemini-3.1-flash-image"
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

model = "gemini-3.7-flash"
n_tokens = count_tokens(model, "The Julia programming language")
println(n_tokens)
```
outputs
```julia
4
```

Pass a `config` to include the system instruction and tools in the count:
```julia
config = GenerateContentConfig(; system_instruction="You are a pirate.", tools=[google_search_tool()])
n_tokens = count_tokens(model, "The Julia programming language"; config)
```

### Create Embeddings

```julia
using GoogleGenAI

embeddings = embed_content("gemini-embedding-001", "Hello")
println(size(embeddings.values))
```
outputs
```julia
(3072,)
```

```julia
using GoogleGenAI

embeddings = embed_content("gemini-embedding-001", ["Hello", "world"])
println(embeddings.response_status)
println(size(embeddings.values[1]))
println(size(embeddings.values[2]))
```
outputs
```julia
200
(3072,)
(3072,)
```

Set what the embedding is for and truncate its size:
```julia
query = embed_content("gemini-embedding-001", "What is Julia?"; task_type="RETRIEVAL_QUERY", output_dimensionality=256)
docs = embed_content(
    "gemini-embedding-001", ["Julia is a programming language."];
    task_type="RETRIEVAL_DOCUMENT", title="Julia", output_dimensionality=256,
)
```

### Decisions

Ask calibrated yes/no, single-choice, and rubric questions about text, images, audio, video, or PDFs.

```julia
using GoogleGenAI

r = create_decision(
    "gemini-decision-1",
    "Third late order and support hung up on me. Cancel my subscription.",
    (
        wants_cancel = NoulQuestion("Does the customer want to cancel?"),
        topic = ChoiceQuestion("Primary topic", ["shipping", "billing", "praise"]),
        sentiment = ScoreQuestion("Customer sentiment", ["negative", "neutral", "positive"]),
    ),
)

r.answers["wants_cancel"].noul        # P(true)
r.answers["topic"].choice             # most likely option
r.answers["topic"].probabilities      # option => probability
r.answers["sentiment"].probabilities  # aligned with r.answers["sentiment"].legend
```

Inputs can mix text and media given as a `path`, raw `data`, a `uri`, or a file from `upload_file`:

```julia
r = create_decision(
    "gemini-decision-1",
    ["Product photo and customer call:", (path="mug.jpg", resolution="high"), upload_file("call.mp3")],
    (damaged = NoulQuestion("Is the product damaged?"),),
)
```

### List Models

`list_models` fetches every page; `get_model` returns one model's details.

```julia
using GoogleGenAI

get_model("gemini-3.7-flash")[:prompt_token_limit]

models = list_models()
for m in models
    if "generateContent" in m[:supported_generation_methods]
        println(m[:name])
    end
end
```
outputs
```julia
...
gemini-3.7-flash
gemini-3.6-flash
gemini-3.5-flash
gemini-3.5-flash-lite
gemini-3.1-flash-lite
gemini-3.1-pro-preview
gemini-3-flash-preview
gemini-2.5-pro
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-2.5-flash-image
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3-pro-image
gemini-3.1-flash-tts-preview
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
gemini-embedding-001
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
model = "gemini-3.7-flash"
prompt = "Hello"
config = GenerateContentConfig(; safety_settings)
response = generate_content(model, prompt; config)
```

### Thinking 

The Gemini 2.5 and 3 series models use an internal "thinking process" during response generation. This process contributes to their improved reasoning capabilities and helps them use multi-step planning to solve complex tasks. Gemini 3 models take a `thinking_level` (`"minimal"`, `"low"`, `"medium"`, `"high"` or `"xhigh"`); a `thinking_budget` in tokens is also accepted and is what Gemini 2.5 models use. Set one or the other; unset fields use the model's default.

```julia
using GoogleGenAI

thinking_config = ThinkingConfig(; thinking_level="low", include_thoughts=true)
config = GenerateContentConfig(; thinking_config)
response = generate_content("gemini-3.7-flash", "Hello"; config)

config = GenerateContentConfig(; thinking_config=ThinkingConfig(; thinking_budget=100))
response = generate_content("gemini-3.7-flash", "Hello"; config)
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
gemini-3.7-flash
gemini-3.6-flash
gemini-3.5-flash
gemini-3.5-flash-lite
gemini-3.1-flash-lite
gemini-3.1-pro-preview
gemini-3-flash-preview
gemini-2.5-pro
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-2.5-flash-image
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3-pro-image
gemini-3.1-flash-tts-preview
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
gemini-embedding-001
gemma-3-1b-it
gemma-3-4b-it
gemma-3-12b-it
gemma-3-27b-it
gemma-3n-e4b-it
gemma-3n-e2b-it
```

Cache content to reuse it across multiple requests:

```julia
using Dates
using GoogleGenAI

# API key is automatically loaded from environment variable
provider = GoogleProvider()
model = "gemini-3.7-flash"

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
# or set an absolute expiry time (UTC)
update_result = update_cached_content(provider, cache_result.name; expire_time=now(UTC) + Hour(1))
# delete a specific cache
delete_cached_content(provider, cache_result.name)
```

Caches can also hold a system instruction and tools, and take a display name:
```julia
cache_result = create_cached_content(
    provider, model, text;
    expire_time=now(UTC) + Minute(30),
    display_name="report",
    system_instruction="Answer questions about the report.",
    tools=[url_context_tool()],
)
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
model = "gemini-3.7-flash"
prompt = "What is this image?"
contents = [prompt, upload_result]
response = generate_content(
    model,
    contents;
)
println(response.text)

# Get file metadata
get_result = get_file(upload_result[:name])

# List all files (every page)
list_result = list_files()

# Delete file
delete_file(upload_result[:name])
```

Files the API generated, such as batch output, can be downloaded with `download_file(name)` (returns the bytes) or `download_file(name; path="out.jsonl")`.

## Chat Sessions

Persistent multi-turn conversations.

```julia
using GoogleGenAI

chat = create_chat("gemini-3.7-flash")
response = send_message(chat, "My name is Luciana.")
println(response.text)

# History is sent automatically on each turn
response = send_message(chat, "What's my name?")

get_history(chat)                 # every turn, including function calls and empty replies
get_history(chat; curated=true)   # what is sent to the model: empty replies and their prompts removed
clear_history!(chat)              # start over
```

Automatic function calling: declare the functions in the config and pass their Julia
implementations when creating the chat; any function calls requested by the model are
executed and fed back until a final answer.

```julia
weather = FunctionDeclaration(
    "get_weather",
    "Get the current weather for a location",
    Dict{String,Any}("type" => "object", "properties" => Dict("location" => Dict("type" => "string"))),
)
functions = Dict{String,Function}(
    "get_weather" => (; location::String) -> Dict("temperature" => 25),
)
config = GenerateContentConfig(; function_declarations=[weather])
chat = create_chat("gemini-3.7-flash"; config, functions)
response = send_message(chat, "What's the weather in Tokyo?")
```

Streaming works the same way:

```julia
for chunk in send_message_stream(chat, "Tell me a story")
    print(chunk.text)
end
```

## Live / Realtime Sessions

Bidirectional streaming over WebSocket (`client.aio.live.connect()` equivalent).

```julia
using GoogleGenAI

# Text-in / text-out turn-based session
session = connect_live(
    "gemini-3.5-flash";
    config=GenerateContentConfig(response_modalities=["TEXT"]),
)
send_client_content(session; turns="Hello!", turn_complete=true)
text = receive_text(session)   # collects model turn text until turnComplete
close_live!(session)
```

Realtime audio/video input and audio output:

```julia
session = connect_live(
    "gemini-2.5-flash-native-audio-preview-09-2025";
    config=GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=Dict(:voiceConfig => Dict(:prebuiltVoiceConfig => Dict(:voiceName => "Kore"))),
    ),
)

# Stream raw 16-bit PCM mono @16kHz (or pass a file path)
send_realtime_input(session; audio=read("input.pcm"), audio_mime_type="audio/pcm;rate=16000")
# Stream camera frames (JPEG bytes or path):
send_realtime_input(session; video=read("frame.jpg"))
# End of an audio stream:
send_realtime_input(session; audio_stream_end=true)

for event in session.events          # or: event = recv_event(session)
    haskey(event, :audio) && play(event[:audio][2])   # (mime_type, PCM bytes)
    haskey(event, :interrupted) && break
end
close_live!(session)
```

Transcripts, resumable sessions, context compression and voice activity detection are set when connecting:

```julia
session = connect_live(
    "gemini-2.5-flash-native-audio-preview-09-2025";
    config=GenerateContentConfig(response_modalities=["AUDIO"]),
    input_audio_transcription=true,    # events carry :input_transcription
    output_audio_transcription=true,   # events carry :output_transcription
    session_resumption=true,           # session.resumption_handle tracks the latest handle
    context_window_compression=true,   # sliding window; or a Dict with triggerTokens/slidingWindow
    realtime_input_config=Dict(:automaticActivityDetection => Dict(:disabled => true)),
)

# Later, resume where the previous session left off:
session = connect_live(model; session_resumption=old_session.resumption_handle)
```

Mid-session tool use: register Julia functions at connect time and any `toolCall`
issued mid-conversation is executed automatically with the result sent back.
Manual replies are also supported via `send_tool_response`.

```julia
functions = Dict{String,Function}(
    "get_weather" => (; city::String) -> Dict("temp_c" => 25),
)
weather = FunctionDeclaration(
    "get_weather",
    "Get the current weather for a city",
    Dict{String,Any}("type" => "object", "properties" => Dict("city" => Dict("type" => "string"))),
)
session = connect_live(provider, "gemini-live-model";
    config=GenerateContentConfig(; function_declarations=[weather]), functions)
```

## Batch Predictions

Submit large volumes of requests asynchronously. Pass inline requests (total size < 20MB) or a JSONL file uploaded via the File API (up to 2GB).

```julia
using GoogleGenAI

# 1. Create an inline batch job
job = create_batch_job("gemini-3.5-flash"; display_name="my-batch", requests=[
    BatchRequest("Tell me a one-sentence joke."; key="request-1"),
    BatchRequest("Why is the sky blue?"; key="request-2"),
])
job_name = String(job[:name])  # e.g. "batches/123456"

# 2. Poll until the job reaches a terminal state
job = poll_batch_job(job_name; interval_s=30, verbose=true)

# 3. Extract results: Vector{Pair{key, text}} (errors reported as "ERROR: ...")
results = extract_batch_text(job)
for (key, text) in results
    println("-- $key --\n$text")
end
```

Each `BatchRequest` takes any `contents` and a `config`; its system instruction, tools, tool config, safety settings and generation settings are all sent:

```julia
config = GenerateContentConfig(; system_instruction="Answer in one word.", tools=[google_search_tool()])
job = create_batch_job("gemini-3.5-flash"; requests=[BatchRequest("Capital of France?"; config)])
```

For larger workloads, upload a JSONL input file first and pass its name:

```julia
# my-batch.jsonl contains one line per request:
# {"key": "request-1", "request": {"contents": [{"parts": [{"text": "..."}]}]}}
file = upload_file("my-batch.jsonl"; mime_type="application/jsonl")
job = create_batch_job("gemini-3.5-flash"; input_file=String(file[:name]))

# After polling succeeds, download and parse the JSONL output file:
finished_job = poll_batch_job(String(job[:name]))
out_file = String(finished_job[:response][:responsesFile])
lines = download_batch_output_file(out_file)
texts = reduce(vcat, extract_batch_text.(lines))
```

Batch embeddings work the same way:

```julia
job = create_embeddings_batch_job("gemini-embedding-001"; contents=["hello", "world"], task_type="CLUSTERING")
```

Other job management functions:
```julia
list_batch_jobs()            # every batch job in the project
get_batch_job(job_name)      # current status/metadata
String(get_batch_job(job_name)[:metadata][:state])
cancel_batch_job(job_name)   # stops processing new requests
delete_batch_job(job_name)   # removes the job entirely
```

## Structured Generation

Use `response_schema` with the API's schema format, or `response_json_schema` with a standard JSON Schema.

```julia
using GoogleGenAI
using JSON3

model = "gemini-3.7-flash"

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

# Built-in Tools

Helpers build the tools the API runs itself: `google_search_tool()`, `url_context_tool()`, `code_execution_tool()`, `google_maps_tool()`, `file_search_tool(store_names)` and `computer_use_tool()`. Raw tool dicts work too.

```julia
config = GenerateContentConfig(; tools=[google_search_tool(), url_context_tool()])
response = generate_content(model, "Summarize https://julialang.org/blog/ in two sentences."; config)
```

# Code Generation

```julia
using GoogleGenAI

config = GenerateContentConfig(; tools=[code_execution_tool()])

model = "gemini-3.7-flash"
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
    "gemini-3.7-flash",
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
    "gemini-3.7-flash",
    conversation_history
)

println("\nFinal response: $(final_response.text)")
```
