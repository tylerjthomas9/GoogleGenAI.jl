 [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
 [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)


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

```julia
using GoogleGenAI

secret_key = ENV["GOOGLE_API_KEY"]
model = "gemini-pro"
prompt = "Hello"
response = generate_content(secret_key, model, prompt)
println(response.text)
```
returns
```julia
"Hello there! How may I assist you today? Feel free to ask me any questions you may have or give me a command. I'm here to help! 😊"
```
