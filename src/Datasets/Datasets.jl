module Datasets
using Random
using DelimitedFiles: readdlm

include("load_datasets.jl")
export load_iris

include("generators.jl")
export make_spiral,
       make_poly,
       make_sin

end