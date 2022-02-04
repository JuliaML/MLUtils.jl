module MLUtils

using Random
using Statistics
using ShowCases: ShowLimit
import StatsBase: sample
using Base: @propagate_inbounds
using Random: AbstractRNG, shuffle!, GLOBAL_RNG

include("observation.jl")
export numobs, 
       getobs, 
       getobs!

include("obstransform.jl")
export mapobs, 
       filterobs, 
       groupobs,
       joinobs,
       shuffleobs
       
include("batchview.jl")
export batchsize,
       batchview, BatchView

include("eachobs.jl")
export eachobs

include("dataloader.jl")
export DataLoader

include("folds.jl")
export kfolds,
       leavepout

include("obsview.jl")
export obsview,
       ObsView

include("randobs.jl")
export randobs

include("resample.jl")
export oversample,
       undersample

include("splitobs.jl")
export splitobs

include("utils.jl")
export batch,
       batchseq,
       chunk,
       flatten,
       group_counts,
       group_indices, 
       normalise,
       stack,
       unbatch,
       unsqueeze,
       unstack
       # rpad

include("Datasets/Datasets.jl")
export Datasets

include("deprecations.jl")

end
