"""
    splitobs(n::Int; at) -> Tuple

Compute the indices for two or more disjoint subsets of
the range `1:n` with splits given by `at`.

# Examples

```julia
julia> splitobs(100, at=0.7)
(1:70, 71:100)

julia> splitobs(100, at=(0.1, 0.4))
(1:10, 11:50, 51:100)
```
"""
splitobs(n::Int; at) = _splitobs(n, at)

_splitobs(n::Int, at::Integer) = _splitobs(n::Int, at / n) 
_splitobs(n::Int, at::NTuple{N, <:Integer}) where {N} = _splitobs(n::Int, at ./ n) 

_splitobs(n::Int, at::Tuple{}) = (1:n,)


function _splitobs(n::Int, at::AbstractFloat)
    0 <= at <= 1 || throw(ArgumentError("the parameter \"at\" must be in interval (0, 1)"))
    n1 = floor(Int, n * at)
    delta = n*at - n1
    # TODO add random rounding
    (1:n1, n1+1:n)
end

function _splitobs(n::Int, at::NTuple{N,<:AbstractFloat}) where N
    at1 = first(at)
    a, b = _splitobs(n::Int, at1)
    n1 = a.stop 
    n2 = b.stop
    at2 = Base.tail(at) .* n ./ (n2 - n1)
    rest = map(x -> n1 .+ x, _splitobs(n2-n1, at2))
    return (a, rest...)
end

"""
    splitobs([rng], data; at, shuffle=false, stratified=nothing) -> Tuple

Partition the `data` into two or more subsets.

The argument `at` specifies how to split the data:
- When `at` is a number between 0 and 1, this specifies the proportion in the first subset.
- When `at` is an integer, it specifies the number of observations in the first subset.
- When `at` is a tuple, entries specifies the number or proportion in each subset, except
for the last which will contain the remaning observations. 
The number of returned subsets is `length(at)+1`.

If `shuffle=true`, randomly permute the observations before splitting.
A random number generator `rng` can be optionally passed as the first argument.

If `stratified` is not `nothing`, it should be an array of labels with the same length as the data.
The observations will be split in a way that the proportion of each label is preserved in each subset.

Supports any datatype implementing [`numobs`](@ref). 

It relies on [`obsview`](@ref) to create views of the data.

# Examples

```jldoctest
julia> splitobs(reshape(1:100, 1, :); at=0.7)  # simple 70%-30% split, of a matrix
([1 2 … 69 70], [71 72 … 99 100])

julia> data = (x=ones(2,10), n=1:10)  # a NamedTuple, consistent last dimension
(x = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], n = 1:10)

julia> splitobs(data, at=(0.5, 0.3))  # a 50%-30%-20% split, e.g. train/test/validation
((x = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], n = 1:5), (x = [1.0 1.0 1.0; 1.0 1.0 1.0], n = 6:8), (x = [1.0 1.0; 1.0 1.0], n = 9:10))

julia> train, test = splitobs((reshape(1.0:100.0, 1, :), 101:200), at=0.7, shuffle=true);  # split a Tuple

julia> vec(test[1]) .+ 100 == test[2]
true
```
"""
splitobs(data; kws...) = splitobs(Random.default_rng(), data; kws...)

function splitobs(rng::AbstractRNG, data; at, 
        shuffle::Bool=false, 
        stratified::Union{Nothing,AbstractVector}=nothing)
    n = numobs(data)
    at = _normalize_at(n, at)
    if shuffle
        perm = randperm(rng, n)
        data = obsview(data, perm) # same as shuffleobs(rng, data), but make it explicit to keep perm
    end
    if stratified !== nothing
        @assert length(stratified) == n
        if shuffle
            stratified = stratified[perm]
        end
        idxs_groups = group_indices(stratified)
        idxs_splits = ntuple(i -> Int[], length(at)+1)
        for (lbl, idxs) in idxs_groups
            new_idxs_splits = splitobs(idxs; at, shuffle=false)
            for i in 1:length(idxs_splits)
                append!(idxs_splits[i], new_idxs_splits[i])
            end
        end
    else
        idxs_splits = splitobs(n; at)
    end
    return map(idxs -> obsview(data, idxs), idxs_splits)
end

_normalize_at(n, at::Integer) = at / n
_normalize_at(n, at::NTuple{N, <:Integer}) where N = at ./ n
_normalize_at(n, at) = at