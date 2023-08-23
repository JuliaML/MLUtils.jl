"""
    numobs(data)

Return the total number of observations contained in `data`.

If `data` does not have `numobs` defined, 
then in the case of `Tables.table(data) == true`
returns the number of rows, otherwise returns `length(data)`.

Authors of custom data containers should implement
`Base.length` for their type instead of `numobs`.
`numobs` should only be implemented for types where there is a
difference between `numobs` and `Base.length`
(such as multi-dimensional arrays).

`getobs` supports by default nested combinations of array, tuple,
named tuples, and dictionaries. 

See also [`getobs`](@ref).

# Examples
```jldoctest

# named tuples 
x = (a = [1, 2, 3], b = rand(6, 3))
numobs(x) == 3

# dictionaries
x = Dict(:a => [1, 2, 3], :b => rand(6, 3))
numobs(x) == 3
```
All internal containers must have the same number of observations:
```juliarepl
julia> x = (a = [1, 2, 3, 4], b = rand(6, 3));

julia> numobs(x)
ERROR: DimensionMismatch: All data containers must have the same number of observations.
Stacktrace:
 [1] _check_numobs_error()
   @ MLUtils ~/.julia/dev/MLUtils/src/observation.jl:163
 [2] _check_numobs
   @ ~/.julia/dev/MLUtils/src/observation.jl:130 [inlined]
 [3] numobs(data::NamedTuple{(:a, :b), Tuple{Vector{Int64}, Matrix{Float64}}})
   @ MLUtils ~/.julia/dev/MLUtils/src/observation.jl:177
 [4] top-level scope
   @ REPL[35]:1
```
"""
function numobs end

# Generic Fallbacks
@traitfn numobs(data::X) where {X; IsTable{X}} = DataAPI.nrow(data)
@traitfn numobs(data::X) where {X; !IsTable{X}} = length(data)


"""
    getobs(data, [idx])

Return the observations corresponding to the observation index `idx`.
Note that `idx` can be any type as long as `data` has defined
`getobs` for that type. If `idx` is not provided, then materialize
all observations in `data`.

If `data` does not have `getobs` defined,
then in the case of `Tables.table(data) == true`
returns the row(s) in position `idx`, otherwise returns `data[idx]`.

Authors of custom data containers should implement
`Base.getindex` for their type instead of `getobs`.
`getobs` should only be implemented for types where there is a
difference between `getobs` and `Base.getindex`
(such as multi-dimensional arrays).

The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision themselves.
The output should be consistent when `idx` is a scalar vs vector.

`getobs` supports by default nested combinations of array, tuple,
named tuples, and dictionaries. 

See also [`getobs!`](@ref) and [`numobs`](@ref).

# Examples

```jldoctest
# named tuples 
x = (a = [1, 2, 3], b = rand(6, 3))

getobs(x, 2) == (a = 2, b = x.b[:, 2])
getobs(x, [1, 3]) == (a = [1, 3], b = x.b[:, [1, 3]])


# dictionaries
x = Dict(:a => [1, 2, 3], :b => rand(6, 3))

getobs(x, 2) == Dict(:a => 2, :b => x[:b][:, 2])
getobs(x, [1, 3]) == Dict(:a => [1, 3], :b => x[:b][:, [1, 3]])
```
"""
function getobs end

# Generic Fallbacks

getobs(data) = data

@traitfn getobs(data::X, idx) where {X; IsTable{X}} = Tables.subset(data, idx, viewhint=false)
@traitfn getobs(data::X, idx) where {X; !IsTable{X}} = data[idx]


"""
    getobs!(buffer, data, idx)

Inplace version of `getobs(data, idx)`. If this method
is defined for the type of `data`, then `buffer` should be used
to store the result, instead of allocating a dedicated object.

Implementing this function is optional. In the case no such
method is provided for the type of `data`, then `buffer` will be
*ignored* and the result of [`getobs`](@ref) returned. This could be
because the type of `data` may not lend itself to the concept
of `copy!`. Thus, supporting a custom `getobs!` is optional
and not required.

See also [`getobs`](@ref) and [`numobs`](@ref). 
"""
function getobs! end
# getobs!(buffer, data) = getobs(data)
getobs!(buffer, data, idx) = getobs(data, idx)

# --------------------------------------------------------------------
# AbstractDataContainer
# Having an AbstractDataContainer allows to define sensible defaults
# for Base (or other) interfaces based on our interface.
# This makes it easier for developers by reducing boilerplate.

abstract type AbstractDataContainer end

Base.size(x::AbstractDataContainer) = (numobs(x),)
Base.iterate(x::AbstractDataContainer, state = 1) =
    (state > numobs(x)) ? nothing : (getobs(x, state), state + 1)
Base.lastindex(x::AbstractDataContainer) = numobs(x)
Base.firstindex(::AbstractDataContainer) = 1

# --------------------------------------------------------------------
# Arrays
# We are very opinionated with arrays: the observation dimension
# is th last dimension. For different behavior wrap the array in 
# a custom type, e.g. with Tables.table.


numobs(A::AbstractArray{<:Any, N}) where {N} = size(A, N)

# 0-dim arrays
numobs(A::AbstractArray{<:Any, 0}) = 1

function getobs(A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    return A[I..., idx]
end

getobs(A::AbstractArray{<:Any, 0}, idx) = A[idx]

function getobs!(buffer::AbstractArray, A::AbstractArray{<:Any, N}, idx) where N
    I = ntuple(_ -> :, N-1)
    buffer .= view(A, I..., idx)
    return buffer
end

function getobs!(buffer::AbstractArray, A::AbstractArray)
    buffer .= A
    return buffer
end

# --------------------------------------------------------------------
# Tuples and NamedTuples

_check_numobs_error() =
    throw(DimensionMismatch("All data containers must have the same number of observations."))

function _check_numobs(data::Union{Tuple, NamedTuple, Dict})
    length(data) == 0 && return 0
    n = numobs(data[first(keys(data))])

    for i in keys(data)
        ni = numobs(data[i])
        n == ni || _check_numobs_error()
    end
    return n
end

numobs(data::Union{Tuple, NamedTuple}) = _check_numobs(data)


getobs(tup::Union{Tuple, NamedTuple}) = map(x -> getobs(x), tup)

Base.@propagate_inbounds function getobs(tup::Union{Tuple, NamedTuple}, indices)
    @boundscheck _check_numobs(tup)
    return map(x -> getobs(x, indices), tup)
end

function getobs!(buffers::Union{Tuple, NamedTuple},
                 tup::Union{Tuple, NamedTuple},
                 indices)
    _check_numobs(tup)

    return map(buffers, tup) do buffer, x
        getobs!(buffer, x, indices)
    end
end

## Dict

numobs(data::Dict) = _check_numobs(data)

getobs(data::Dict, i) = Dict(k => getobs(v, i) for (k, v) in pairs(data))

getobs(data::Dict) = Dict(k => getobs(v) for (k, v) in pairs(data))

function getobs!(buffers, data::Dict, i)
    for (k, v) in pairs(data)
        getobs!(buffers[k], v, i)
    end

    return buffers
end


