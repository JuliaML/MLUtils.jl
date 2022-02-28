

"""
    collate(samples)

Collates a vector of observations into a single batch. Vectors of
arrays are concatenated by adding a batch dimension.
Other types are pivoted recursively, e.g a `Vector` of `Tuple`s
is turned into a `Tuple` of `Vector`s.

## Examples

```julia
julia> collate([
        [1, 2],
        [3, 4],
    ])
2×2 Matrix{Int64}:
 1  3
 2  4
```

```julia
julia> collate([Dict(:x => 1), Dict(:x => 2)])
Dict{Symbol, Vector{Int64}} with 1 entry:
    :x => [1, 2]
```
"""
function collate(samples::AbstractVector{<:Tuple})
    !isempty(samples) || return samples
    return tuple([collate([s[key] for s in samples]) for key in keys(samples[1])]...)
end

function collate(samples::AbstractVector{<:Dict})
    !isempty(samples) || return samples
    return Dict(key => collate([s[key] for s in samples]) for key in keys(samples[1]))
end


# base case: vector of arrays -> batched array
function collate(obss::AbstractVector{<:AbstractArray{T, N}}) where {T, N}
    reduce((a1, a2) -> cat(a1, a2; dims = N + 1), obss)
end

# vector of tuples -> tuple of collated observations
function collate(samples::AbstractVector{<:Tuple})
    !isempty(samples) || return samples
    return tuple([collate([s[key] for s in samples]) for key in keys(samples[1])]...)
end

# vector of named tuples -> named tuple of collated observations
function collate(samples::AbstractVector{<:NamedTuple})
    !isempty(samples) || return samples
    return NamedTuple((key => collate([s[key] for s in samples]) for key in keys(samples[1])))
end

# fallback does nothing
collate(obss) = obss
