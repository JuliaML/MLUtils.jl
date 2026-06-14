"""
    randobs([rng], data, [n])

Pick a random observation or a batch of `n` random observations
from `data`.
For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref).

A random number generator `rng` can be optionally passed as the
first argument.
"""
randobs(data) = randobs(Random.default_rng(), data)
randobs(rng::AbstractRNG, data) = getobs(data, rand(rng, 1:numobs(data)))

randobs(data, n) = randobs(Random.default_rng(), data, n)
randobs(rng::AbstractRNG, data, n) = getobs(data, rand(rng, 1:numobs(data), n))
