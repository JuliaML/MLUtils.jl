using MLUtils
using Documenter

# Copy the README to the home page in docs, to avoid duplication.
readme = readlines(joinpath(@__DIR__, "..", "README.md"))

open(joinpath(@__DIR__, "src/index.md"), "w") do f
    for l in readme
        println(f, l)
    end
end

DocMeta.setdocmeta!(MLUtils, :DocTestSetup, :(using MLUtils); recursive=true)

makedocs(;
    modules=[MLUtils],
    doctest=true, 
    clean=true,     
    sitename = "MLUtils.jl",
    pages = ["Home" => "index.md",
             "API" => "api.md"],
)

deploydocs(repo="github.com/JuliaML/MLUtils.jl.git", 
          devbranch="main")
