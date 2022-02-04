"""
    splitobs(n::Int; at) -> Tuple

Pre-compute the indices for two disjoint subsets and return them
as a tuple of two ranges. The first range will span the first
`at` fraction of possible indices, while the second range will
cover the rest. These indices are applicable to any data
container of size `n`.

```julia
julia> splitobs(100, at=0.7)
(1:70, 71:100)
```

A tuple `at` can be passed for multiple splits:

```julia    
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
    splitobs(data; at) -> Tuple

Split the `data` into multiple subsets proportional to the
value(s) of `at`.

Note that this function will perform the splits statically and
thus not perform any randomization. The function creates a
`NTuple` of data subsets in which the first N-1 elements/subsets
contain the fraction of observations of `data` that is specified
by `at`.

For example, if `at` is a `Float64` then the return-value will be
a tuple with two elements (i.e. subsets), in which the first
element contains the fracion of observations specified by `at`
and the second element contains the rest. In the following code
the first subset `train` will contain the first 70% of the
observations and the second subset `test` the rest.

```julia
train, test = splitobs(X, at=0.7)
```

If `at` is a tuple of `Float64` then additional subsets will be
created. In this example `train` will have the first 50% of the
observations, `val` will have next 30%, and `test` the last 20%

```julia
train, val, test = splitobs(X, at=(0.5, 0.3))
```

It is also possible to call `splitobs` with multiple data
arguments as tuple, which all must have the same number of total
observations. This is useful for labeled data.

```julia
train, test = splitobs((X, y), at=0.7)
(x_train,y_train), (x_test,y_test) = splitobs((X, y), at=0.7)
```

If the observations should be randomly assigned to a subset,
then you can combine the function with `shuffleobs`

```julia
# This time observations are randomly assigned.
train, test = splitobs(shuffleobs((X, y)), at=0.7)
```

See [`stratifiedobs`](@ref) for a related function that preserves
the target distribution.
"""
function splitobs(data; at)
    n = numobs(data)
    map(idx -> obsview(data, idx), splitobs(n; at))
end
