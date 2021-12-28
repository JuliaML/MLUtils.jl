"""
    abstract DataView{TElem, TData} <: AbstractVector{TElem}

Baseclass for all vector-like views of some data structure.
This allow for example to see some design matrix as a vector of
individual observation-vectors instead of one matrix.
see `ObsView` and `BatchView` for examples.
"""
abstract type DataView{TElem, TData} <: AbstractVector{TElem} end

Base.IndexStyle(::Type{T}) where {T<:DataView} = IndexLinear()
Base.size(A::DataView) = (length(A),)
Base.lastindex(A::DataView) = length(A)
getobs(A::DataView) = map(getobs, A)
getobs(A::DataView, i) = getobs(A[i])


# --------------------------------------------------------------------

"""
    ObsView(data)

Description
============

Create a view of the given `data` in the form of a vector of
individual observations. Any data access is delayed until
`getindex` is called, and even `getindex` returns the result of
[`datasubset`](@ref) which in general avoids data movement until
[`getobs`](@ref) is called.

If used as an iterator, the view will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
lazy subset to the current observation.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).


Methods
========

Aside from the `AbstractArray` interface following additional
methods are provided:

- **`getobs(data::ObsView, indices::AbstractVector)`** :
    Returns a `Vector` of indivdual observations specified by
    `indices`.

- **`numobs(data::ObsView)`** :
    Returns the number of observations in `data` that the
    iterator will go over.

Details
========

For `ObsView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See `?DataSubset` for more info.

Examples
=========

```julia
X, Y = MLDataUtils.load_iris()

A = obsview(X)
@assert typeof(A) <: ObsView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,1}
@assert length(A) == 150 # Iris has 150 observations
@assert size(A[1]) == (4,) # Iris has 4 features

for x in obsview(X)
    @assert typeof(x) <: SubArray{Float64,1}
end

# iterate over each individual labeled observation
for (x,y) in obsview((X,Y))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# same but in random order
for (x,y) in obsview(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

[`eachobs`](@ref), [`BatchView`](@ref), [`shuffleobs`](@ref),
[`getobs`](@ref), [`numobs`](@ref), [`DataSubset`](@ref)
"""
struct ObsView{TElem,TData} <: DataView{TElem,TData}
    data::TData
end

const obsview = ObsView

function ObsView(data::T) where {T}
    E = typeof(datasubset(data, 1))
    ObsView{E,T}(data)
end

function ObsView(A::T) where T<:DataView
    @warn string("Trying to nest a ", T.name, " into an ObsView, which is not supported. Returning ObsView(parent(_)) instead")
    ObsView(parent(A))
end

ObsView(A::ObsView) = A

numobs(A::ObsView) = numobs(A.data)
Base.parent(A::ObsView) = A.data
Base.length(A::ObsView) = numobs(A)
Base.getindex(A::ObsView, i::Int) = datasubset(A.data, i)
Base.getindex(A::ObsView, i::AbstractVector) = ObsView(datasubset(A.data, i))

function Base.showarg(io::IO, A::ObsView, toplevel)
    print(io, "obsview(")
    Base.showarg(io, parent(A), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simplify
end

# ------------------------------------------------------------


# --------------------------------------------------------------------

# # if subsetting a DataView, then DataView the subset instead.
# for fun in (:DataSubset, :datasubset)
#     @eval @generated function ($fun)(A::T, i, obsdim::$O) where T<:AbstractObsView
#         quote
#             @assert obsdim == A.obsdim
#             ($(T.name.name))(($($fun))(parent(A), i, obsdim), obsdim)
#         end
#     end
#     @eval function ($fun)(A::BatchView, i, obsdim::$O)
#         @assert obsdim == A.obsdim
#         length(i) < A.size && throw(ArgumentError("The chosen batch-size ($(A.size)) is greater than the number of observations ($(length(i)))"))
#         BatchView(($fun)(parent(A), i, obsdim), A.size, -1, obsdim)
#     end
# end

DataSubset(A::ObsView, i) = ObsView(DataSubset(parent(A), i))
datasubset(A::ObsView, i) = ObsView(datasubset(parent(A), i))
