# TODO: allow passing a rng as first parameter

"""
    randobs(data, [n])

Pick a random observation or a batch of `n` random observations
from `data`.
For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref).
"""
randobs(data) = getobs(data, rand(1:numobs(data)))

randobs(data, n) = getobs(data, rand(1:numobs(data), n))