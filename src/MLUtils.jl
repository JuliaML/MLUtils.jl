module MLUtils
using Random

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
export DataView, ObsView

end
