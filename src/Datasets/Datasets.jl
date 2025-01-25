module Datasets
using Random
using DelimitedFiles: readdlm
using MLUtils: getobs, shuffleobs

include("load_datasets.jl")
export load_iris

include("generators.jl")
export make_spiral,
       make_poly,
       make_sin,
       make_moons

end