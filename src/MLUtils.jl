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
       joinobs
       
include("batchview.jl")
export batchsize,
       batchview, BatchView

include("dataiterator.jl")
export eachobs, 
       eachbatch

include("dataloader.jl")
export DataLoader

include("folds.jl")
export kfolds,
       leavepout

include("obsview.jl")
export datasubset,
       obsview, ObsView

include("randobs.jl")
export randobs

include("resample.jl")
export labelmap, 
       oversample,
       undersample

include("shuffleobs.jl")
export shuffleobs

include("splitobs.jl")
export splitobs

include("utils.jl")
export batch,
       batchseq,
       chunk,
       flatten,
       frequencies,
       normalise,
       stack,
       unbatch,
       unsqueeze,
       unstack
       # rpad

include("Datasets/Datasets.jl")
export Datasets

end
