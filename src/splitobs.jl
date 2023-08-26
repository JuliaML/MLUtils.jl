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

Partition the `data` into two or more subsets.
When `at` is a number (between 0 and 1) this specifies the proportion in the first subset.
When `at::Tuple`, each entry specifies the proportion an a subset, with the last having `1-sum(ast)`.

If `shuffle=true`, randomly permute the observations before splitting.

Supports any datatype implementing the [`numobs`](@ref) and
[`getobs`](@ref) interfaces -- including arrays, tuples & NamedTuples of arrays.

# Examples

```jldoctest
julia> splitobs(permutedims(1:100); at=0.7)  # simple 70%-30% split, of a matrix
([1 2 … 69 70], [71 72 … 99 100])

julia> data = (x=ones(2,10), n=1:10)  # a NamedTuple, consistent last dimension
(x = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], n = 1:10)

julia> splitobs(data, at=(0.5, 0.3))  # a 50%-30%-20% split, e.g. train/test/validation
((x = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], n = 1:5), (x = [1.0 1.0 1.0; 1.0 1.0 1.0], n = 6:8), (x = [1.0 1.0; 1.0 1.0], n = 9:10))

julia> train, test = splitobs((permutedims(1.0:100.0), 101:200), at=0.7, shuffle=true);  # split a Tuple

julia> vec(test[1]) .+ 100 == test[2]
true
```
"""
function splitobs(data; at, shuffle::Bool=false)
    if shuffle
        data = shuffleobs(data)
    end
    n = numobs(data)
    return map(idx -> obsview(data, idx), splitobs(n; at))
end
