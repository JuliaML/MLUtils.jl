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
    n1 = clamp(round(Int, at*n), 0, n)
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
    splitobs(data; at, shuffle=false) -> Tuple

Split the `data` into multiple subsets proportional to the
value(s) of `at`. 

If `shuffle=true`, randomly permute the observations before splitting.

Supports any datatype implementing the [`numobs`](@ref) and
[`getobs`](@ref) interfaces.

# Examples

```julia
# A 70%-30% split
train, test = splitobs(X, at=0.7)

# A 50%-30%-20% split
train, val, test = splitobs(X, at=(0.5, 0.3))

# A 70%-30% split with multiple arrays and shuffling
train, test = splitobs((X, y), at=0.7, shuffle=true)
Xtrain, Ytrain = train
```
"""
function splitobs(data; at, shuffle=false)
    shuffle && return splitobs(shuffleobs(data); at, shuffle=false)
    n = numobs(data)
    return map(idx -> obsview(data, idx), splitobs(n; at))
end
