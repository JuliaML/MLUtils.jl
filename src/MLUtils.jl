module MLUtils
using Random
import StatsBase: sample

include("utils.jl")

include("observation.jl")
export numobs, getobs, getobs!

include("randobs.jl")
export randobs

include("obsview.jl")
export obsview, ObsView,
       datasubset

include("shuffleobs.jl")
export shuffleobs

include("splitobs.jl")
export splitobs

include("batchview.jl")
export batchview, BatchView, 
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
