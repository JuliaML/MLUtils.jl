"""
    ObsView(data, [indices])

Description
============

Used to represent a subset of some `data` of arbitrary type by
storing which observation-indices the subset spans. Furthermore,
subsequent subsettings are accumulated without needing to access
actual data.

The main purpose for the existence of `ObsView` is to delay
data access and movement until an actual batch of data (or single
observation) is needed for some computation. This is particularily
useful when the data is not located in memory, but on the hard
drive or some remote location. In such a scenario one wants to
load the required data only when needed.

This type is usually not constructed manually, but instead
instantiated by calling [`obsview`](@ref),
[`shuffleobs`](@ref), or [`splitobs`](@ref)

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).

- **`indices`** : Optional. The index or indices of the
    observation(s) in `data` that the subset should represent.
    Can be of type `Int` or some subtype of `AbstractVector`.

Methods
========

- **`getindex`** : Returns the observation(s) of the given
    index/indices as a new `ObsView`. No data is copied aside
    from the required indices.

- **`numobs`** : Returns the total number observations in the subset
    (**not** the whole data set underneath).

- **`getobs`** : Returns the underlying data that the
    `ObsView` represents at the given relative indices. Note
    that these indices are in "subset space", and in general will
    not directly correspond to the same indices in the underlying
    data set.

Details
========

For `ObsView` to work on some data structure, the desired type
`MyType` must implement the following interface:

- `getobs(data::MyType, idx)` :
    Should return the observation(s) indexed by `idx`.
    In what form is up to the user.
    Note that `idx` can be of type `Int` or `AbstractVector`.

- `numobs(data::MyType)` :
    Should return the total number of observations in `data`

The following methods can also be provided and are optional:

- `getobs(data::MyType)` :
    By default this function is the identity function.
    If that is not the behaviour that you want for your type,
    you need to provide this method as well.

- `datasubset(data::MyType, idx)` :
    If your custom type has its own kind of subset type, you can
    return it here. An example for such a case are `SubArray` for
    representing a subset of some `AbstractArray`.
   
- `getobs!(buffer, data::MyType, [idx])` :
    Inplace version of `getobs(data, idx)`. If this method
    is provided for `MyType`, then `eachobs` and `eachbatch`
    (among others) can preallocate a buffer that is then reused
    every iteration. Note: `buffer` should be equivalent to the
    return value of `getobs(::MyType, ...)`, since this is how
    `buffer` is preallocated by default.


Examples
=========

```julia
X, y = MLUtils.load_iris()

# The iris set has 150 observations and 4 features
@assert size(X) == (4,150)

# Represents the 80 observations as a ObsView
subset = ObsView(X, 21:100)
@assert numobs(subset) == 80
@assert typeof(subset) <: ObsView
# getobs indexes into the subset
@assert getobs(subset, 1:10) == X[:, 21:30]

# The lowercase version tries to avoid boxing into ObsView
# for types that provide a custom "subset", such as arrays.
# Here it instead creates a native SubArray.
subset = obsview(X, 1:100)
@assert numobs(subset) == 100
@assert typeof(subset) <: SubArray

# Also works for tuples of arbitrary length
subset = obsview((X,y), 1:100)
@assert numobs(subset) == 100
@assert typeof(subset) <: Tuple # tuple of SubArray

# `subset` also works for tuple of data. (useful for labeled data)
subset = obsview((X, y), 1:100)
@assert numobs(subset) == 100
@assert typeof(subset) <: Tuple # Tuple of ObsView

# Split dataset into training and test split
train, test = splitobs(shuffleobs((X, y)), at=0.7)
@assert typeof(train) <: Tuple # of SubArray
@assert typeof(test)  <: Tuple # of SubArray
@assert numobs(train) == 105
@assert numobs(test) == 45
```

see also
=========

[`obsview`](@ref),  [`getobs`](@ref), [`numobs`](@ref),
[`splitobs`](@ref), [`shuffleobs`](@ref),
[`kfolds`](@ref), [`batchview`](@ref).
"""
struct ObsView{Tdata, I<:Union{Int,AbstractVector}} <: AbstractDataContainer
    data::Tdata
    indices::I

    function ObsView(data::T, indices::I) where {T,I}
        1 <= minimum(indices) || throw(BoundsError(data, indices))
        maximum(indices) <= numobs(data) || throw(BoundsError(data, indices))
        new{T,I}(data, indices)
    end
end

ObsView(data) = ObsView(data, 1:numobs(data))

# # don't nest subsets
ObsView(subset::ObsView) = subset

function ObsView(subset::ObsView, indices::Union{Int,AbstractVector})
    ObsView(subset.data, subset.indices[indices])
end

function Base.show(io::IO, subset::ObsView)
    if get(io, :compact, false)
        print(io, "ObsView{", typeof(subset.data), "} with " , numobs(subset), " observations")
    else
        print(io, summary(subset), "\n ", numobs(subset), " observations")
    end
end

function Base.summary(subset::ObsView)
    io = IOBuffer()
    print(io, typeof(subset).name.name, "(")
    Base.showarg(io, subset.data, false)
    print(io, ", ")
    Base.showarg(io, subset.indices, false)
    print(io, ')')
    first(readlines(seek(io,0)))
end

# compare if both subsets cover the same observations of the same data
# we don't care how the indices are stored, just that they match
# in order and values
function Base.:(==)(s1::ObsView, s2::ObsView)
    s1.data == s2.data && s1.indices == s2.indices
end

Base.IteratorEltype(::Type{<:ObsView}) = Base.EltypeUnknown()

# override AbstractDataContainer defaults
Base.getindex(subset::ObsView, idx) =
    datasubset(subset.data, subset.indices[idx])

numobs(subset::ObsView) = length(subset.indices)

getobs(subset::ObsView) = getobs(subset.data, subset.indices)
getobs(subset::ObsView, idx) = getobs(subset.data, subset.indices[idx])

getobs!(buffer, subset::ObsView) = getobs!(buffer, subset.data, subset.indices)
getobs!(buffer, subset::ObsView, idx) = getobs!(buffer, subset.data, subset.indices[idx])

Base.parent(x::ObsView) = x.data

@doc (@doc ObsView)
const obsview = ObsView

# --------------------------------------------------------------------

"""
    datasubset(data, [indices])

Returns a lazy view of the observations in `data` that
correspond to the given `indices`. No data will be copied except
of the indices. It is similar to calling `ObsView(data,
[indices])`, but returns a `SubArray` if the type of
`data` is `Array` or `SubArray`. Furthermore, this function may
be extended for custom types of `data` that also want to provide
their own subset-type.

In case `data` is some `Tuple`, the constructor will be mapped
over its elements. That means that the constructor returns a
`Tuple` of `ObsView` instead of a `ObsView` of `Tuple`.

If instead you want to get the subset of observations
corresponding to the given `indices` in their native type, use
`getobs`.

see `ObsView` for more information.
"""
datasubset(data, indices=1:numobs(data)) = ObsView(data, indices)

##### Arrays / SubArrays

datasubset(A::SubArray) = A

function datasubset(A::AbstractArray{T,N}, idx) where {T,N}
    I = ntuple(_ -> :, N-1)
    return view(A, I..., idx)
end

getobs(a::SubArray) = getobs(a.parent, last(a.indices))

##### Tuples / NamedTuples
function datasubset(tup::Union{Tuple, NamedTuple}, indices)
    map(data -> datasubset(data, indices), tup)
end
