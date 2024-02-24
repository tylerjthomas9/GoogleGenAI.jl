using GoogleGenAI
using Documenter

DocMeta.setdocmeta!(GoogleGenAI, :DocTestSetup, :(using GoogleGenAI); recursive=true)

makedocs(;
    modules=[GoogleGenAI],
    authors="Tyler Thomas <tylerjthomas9@gmail.com>",
    repo="https://github.com/tylerjthomas9/GoogleGenAI.jl.git",
    sitename="GoogleGenAI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://tylerjthomas9.github.io/GoogleGenAI.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md", "API" => "api.md"],
    warnonly=true,
)

deploydocs(; repo="github.com/tylerjthomas9/GoogleGenAI.jl.git")
