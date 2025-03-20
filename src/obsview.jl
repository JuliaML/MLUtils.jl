"""
    ObsView(data, [indices])

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

Any data access is delayed until `getindex` is called, 
and even `getindex` returns the result of
[`obsview`](@ref) which in general avoids data movement until
[`getobs`](@ref) is called.
If used as an iterator, the view will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
lazy subset to the current observation.

# Arguments

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).

- **`indices`** : Optional. The index or indices of the
    observation(s) in `data` that the subset should represent.
    Can be of type `Int` or some subtype of `AbstractVector`.

# Methods

- **`getindex`** : Returns the observation(s) of the given
    index/indices. No data is copied aside
    from the required indices.

- **`numobs`** : Returns the total number observations in the subset.

- **`getobs`** : Returns the underlying data that the
    `ObsView` represents at the given relative indices. Note
    that these indices are in "subset space", and in general will
    not directly correspond to the same indices in the underlying
    data set.

# Details

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

- `obsview(data::MyType, idx)` :
    If your custom type has its own kind of subset type, you can
    return it here. An example for such a case are `SubArray` for
    representing a subset of some `AbstractArray`.

- `getobs!(buffer, data::MyType, [idx])` :
    Inplace version of `getobs(data, idx)`. If this method
    is provided for `MyType`, then `eachobs` can preallocate a buffer that is then reused
    every iteration. Note: `buffer` should be equivalent to the
    return value of `getobs(::MyType, ...)`, since this is how
    `buffer` is preallocated by default.

# Examples

```julia
X, Y = MLUtils.load_iris()

# The iris set has 150 observations and 4 features
@assert size(X) == (4,150)

# Represents the 80 observations as a ObsView
v = ObsView(X, 21:100)
@assert numobs(v) == 80
@assert typeof(v) <: ObsView
# getobs indexes into v
@assert getobs(v, 1:10) == X[:, 21:30]

# Use `obsview` to avoid boxing into ObsView
# for types that provide a custom "subset", such as arrays.
# Here it instead creates a native SubArray.
v = obsview(X, 1:100)
@assert numobs(v) == 100
@assert typeof(v) <: SubArray

# Also works for tuples of arbitrary length
subset = obsview((X, Y), 1:100)
@assert numobs(subset) == 100
@assert typeof(subset) <: Tuple # tuple of SubArray

# Use as iterator
for x in ObsView(X)
    @assert typeof(x) <: SubArray{Float64,1}
end

# iterate over each individual labeled observation
for (x, y) in ObsView((X, Y))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# same but in random order
for (x, y) in ObsView(shuffleobs((X, Y)))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# Indexing: take first 10 observations
x, y = ObsView((X, Y))[1:10]
```

# See also

[`obsview`](@ref),  [`getobs`](@ref), [`numobs`](@ref),
[`splitobs`](@ref), [`shuffleobs`](@ref),
[`kfolds`](@ref).
"""
struct ObsView{Tdata, I<:Union{Int,AbstractVector}} <: AbstractDataContainer
    data::Tdata
    indices::I

    function ObsView(data::T, indices::I) where {T,I}
        if !isempty(indices)
            1 <= minimum(indices) || throw(BoundsError(data, indices))
            maximum(indices) <= numobs(data) || throw(BoundsError(data, indices))
        end
        return new{T,I}(data, indices)
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

@propagate_inbounds Base.getindex(subset::ObsView, idx) =
    obsview(subset.data, subset.indices[idx])

Base.length(subset::ObsView) = length(subset.indices)

getobs(subset::ObsView) = getobs(subset.data, subset.indices)
@propagate_inbounds getobs(subset::ObsView, idx) = getobs(subset.data, subset.indices[idx])

getobs!(buffer, subset::ObsView) = getobs!(buffer, subset.data, subset.indices)
@propagate_inbounds getobs!(buffer, subset::ObsView, idx) = getobs!(buffer, subset.data, subset.indices[idx])

Base.parent(x::ObsView) = x.data


# --------------------------------------------------------------------

"""
    obsview(data, [indices])

Return a lazy view of the observations in `data` that
correspond to the given `indices`. No data will be copied. 

By default the return is an [`ObsView`](@ref), although this can be
overloaded for custom types of `data` that want to provide
their own lazy view.

In case `data` is a tuple or named tuple, the constructor will be mapped
over its elements. For array types, return a subarray.

The observation in the returned view `ov` can be materialized by calling
`getobs(ov, i)` on the view, where `i` is an index in `1:length(ov)`.

If `indices` is not provided, it will be assumed to be `1:numobs(data)`.

# Examples

```jldoctest
julia> obsview([1 2 3; 4 5 6], 1:2)
2×2 view(::Matrix{Int64}, :, 1:2) with eltype Int64:
 1  2
 4  5
```
"""
obsview(data, indices) = ObsView(data, indices)
obsview(data) = obsview(data, 1:numobs(data))

##### Arrays / SubArrays

obsview(A::SubArray) = A

"""
    obsview(data::AbstractArray, [obsdim])
    obsview(data::AbstractArray, idxs, [obsdim])

Return a view of the array `data` that correspond to the given
indices `idxs`. If `obsdim` of type [`ObsDim`](@ref) is provided, the observation 
dimension of the array is assumed to be along that dimension, otherwise
it is assumed to be the last dimension.

If `idxs` is not provided, it will be assumed to be `1:numobs(data)`.

# Examples

```jldoctest
julia> x = rand(4, 5, 2);

julia> v = obsview(x, 2:3, ObsDim(2));

julia> numobs(v)
2

julia> getobs(v, 1) == x[:, 2, :]
true

julia> getobs(v, 1:2) == x[:, 2:3, :]
true
```
"""
obsview(data::AbstractArray) = obsview(data, 1:numobs(data))

function obsview(A::AbstractArray{T,N}, idx) where {T,N}
    I = ntuple(_ -> :, N-1)
    return view(A, I..., idx)
end

getobs(a::SubArray) = getobs(a.parent, last(a.indices))

### Arrays + ObsDim

"""
    ObsDim(d::Int)

Type to specify the observation dimension of an array.

It can be used in combination with [`obsview`](@ref).
"""
struct ObsDim{D} end

ObsDim(d::Int) = ObsDim{d}()
ObsDim(obsdim::ObsDim) = obsdim

function obsview(data::A, ::ObsDim{D}) where {A<:AbstractArray,D}
    idx = 1:size(data, D)
    return ArrayObsView{D,A,typeof(idx)}(data, idx)
end

function obsview(data::A, idx, ::ObsDim{D}) where {A<:AbstractArray,D}
    return ArrayObsView{D,A,typeof(idx)}(data, idx)
end

struct ArrayObsView{ObsDim,A<:AbstractArray,I<:AbstractVector} <: AbstractDataContainer
    data::A
    indices::I
end

Base.length(x::ArrayObsView) = length(x.indices)

function Base.getindex(x::ArrayObsView{ObsDim}, idx) where {ObsDim}
    # return a view, consistently with ObsView behaviour
    return selectdim(x.data, ObsDim, x.indices[idx])
end

function getobs(x::ArrayObsView{ObsDim,A}, idx) where {ObsDim, T, N, A<:AbstractArray{T,N}}
    Ipre = ntuple(_ -> :, ObsDim-1)
    Ipost = ntuple(_ -> :, N-ObsDim)
    return x.data[Ipre..., x.indices[idx], Ipost...]
end

getobs(x::ArrayObsView) = getobs(x, 1:length(x))

function Base.show(io::IO, x::ArrayObsView{ObsDim}) where {ObsDim}
    print(io, "ArrayObsView($(summary(x.data)), obsdim=$(ObsDim), numobs=$(length(x)))")
end

##### Tuples / NamedTuples
function obsview(tup::Union{Tuple, NamedTuple}, indices)
    return map(data -> obsview(data, indices), tup)
end


