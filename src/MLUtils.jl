module MLUtils


include("observation.jl")
export numobs, getobs, getobs!

include("randobs.jl")
export randobs

end
