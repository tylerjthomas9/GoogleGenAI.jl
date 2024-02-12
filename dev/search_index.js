var documenterSearchIndex = {"docs":
[{"location":"#GoogleGenAI.jl-Docs","page":"Home","title":"GoogleGenAI.jl Docs","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Julia wrapper to the Google generative AI API. For API functionality, see reference documentation.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"From source:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(url=\"https://github.com/tylerjthomas9/GoogleGenAI.jl/\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]  # enters the pkg interface\nPkg> add https://github.com/tylerjthomas9/GoogleGenAI.jl/","category":"page"},{"location":"#Quick-Start","page":"Home","title":"Quick Start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Create a secret API key in Google AI Studio","category":"page"},{"location":"","page":"Home","title":"Home","text":"using GoogleGenAI\n\nsecret_key = ENV[\"GOOGLE_API_KEY\"]\nmodel = \"gemini-pro\"\nprompt = \"Hello\"\nresponse = generate_content(secret_key, model, prompt)\nprintln(response.text)","category":"page"},{"location":"","page":"Home","title":"Home","text":"returns","category":"page"},{"location":"","page":"Home","title":"Home","text":"\"Hello there! How may I assist you today? Feel free to ask me any questions you may have or give me a command. I'm here to help! 😊\"","category":"page"}]
}
