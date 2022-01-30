module Datasets
using Random
using DelimitedFiles: readdlm

include("datasets.jl")
export load_iris,
       load_line,
       load_sin,
       load_spiral,
       load_poly

include("generators.jl")
export make_spiral,
       make_poly,
       make_sin

end