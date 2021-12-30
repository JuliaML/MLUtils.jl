"""
    shuffleobs([rng], data)

Return a "subset" of `data` that spans all observations, but
has the order of the observations shuffled.

The values of `data` itself are not copied. Instead only the
indices are shuffled. This function calls [`obsview`](@ref) to
accomplish that, which means that the return value is likely of a
different type than `data`.

```julia
# For Arrays the subset will be of type SubArray
@assert typeof(shuffleobs(rand(4,10))) <: SubArray

# Iterate through all observations in random order
for x in eachobs(shuffleobs(X))
    ...
end
```

The optional parameter `rng` allows one to specify the
random number generator used for shuffling. This is useful when
reproducible results are desired. By default, uses the global RNG.
See `Random` in Julia's standard library for more info.

For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref). See [`ObsView`](@ref)
for more information.
"""
shuffleobs(data) = shuffleobs(Random.GLOBAL_RNG, data)

function shuffleobs(rng::AbstractRNG, data)
    datasubset(data, randperm(rng, numobs(data)))
end
