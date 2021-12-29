module MLUtils
using Random
import StatsBase: sample

include("utils.jl")

include("observation.jl")
export numobs, getobs, getobs!

include("randobs.jl")
export randobs

include("datasubset.jl")
export datasubset, DataSubset

include("shuffleobs.jl")
export shuffleobs

include("splitobs.jl")
export splitobs

include("dataview.jl")
export DataView,
       obsview, ObsView,
       batchview, BatchView, 
       batchsize

include("dataiterator.jl")
export eachobs, 
       eachbatch

include("folds.jl")
export kfolds,
       leavepout

include("resample.jl")
export labelmap, 
        oversample,
        undersample

end
