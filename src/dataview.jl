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
    Returns a `Vector` of individual observations specified by
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
X, Y = MLUtils.load_iris()

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

datasubset(A::ObsView, i) = A[i]

function Base.showarg(io::IO, A::ObsView, toplevel)
    print(io, "obsview(")
    Base.showarg(io, parent(A), false)
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simplify
end

# --------------------------------------------------------------------

"""
    BatchView(data, size; partial=true)
    BatchView(data; size=1, partial=true)


Description
============

Create a view of the given `data` that represents it as a vector
of batches. Each batch will contain an equal amount of
observations in them. The batch-size
can be specified using the  parameter `size`.
In the case that the size of the dataset is not dividable by the
specified `size`, the remaining observations will
be ignored if `partial=false`. If  `partial=true` instead 
the last batch size can be slight smaller.

Note that any data access is delayed until `getindex` is called,
and even `getindex` returns the result of [`datasubset`](@ref)
which in general avoids data movement until [`getobs`](@ref) is
called.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. 

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).

- **`size`** : The batch-size of each batch. I.e. the number of
    observations that each batch must contain.

- **`partial`** : If `partial=false` and the number of observations is 
    not divisible by the batch size, then the last mini-batch is dropped.

Details
========

For `BatchView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See `?DataSubset` for more info.


Examples
=========

```julia
using MLUtils
X, Y = MLUtils.load_iris()

A = batchview(X, size = 30)
@assert typeof(A) <: BatchView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,2}
@assert length(A) == 5 # Iris has 150 observations
@assert size(A[1]) == (4,30) # Iris has 4 features

# 5 batches of size 30 observations
for x in batchview(X, size = 30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert numobs(x) === 30
end

# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x,y) in batchview((X,Y), size = 20, partial=false)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert numobs(x) == numobs(y) == 20
end


# randomly assign observations to one and only one batch.
for (x,y) in batchview(shuffleobs((X,Y)), size=20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end
```

see also
=========

[`eachbatch`](@ref), [`ObsView`](@ref), [`shuffleobs`](@ref),
[`getobs`](@ref), [`numobs`](@ref), [`DataSubset`](@ref)
"""
struct BatchView{TElem,TData} <: DataView{TElem,TData}
    data::TData
    size::Int
    count::Int
    partial::Bool
    imax::Int
end

function BatchView(data::T; size::Int=1, partial::Bool=true) where {T}
    # nsize, ncount = _compute_batch_settings(data, size, count, upto)
    n = numobs(data)
    if n < size
        @warn "Number of observations less than batch size, decreasing the batch size to $n"
        size = n
    end
    E = typeof(datasubset(data, 1:size))
    imax = partial ? n : n - size + 1
    count = partial ? ceil(Int, n / size) : floor(Int, n / size)
    BatchView{E,T}(data, size, count, partial, imax)
end

"""
    batchsize(data) -> Int

Return the fixed size of each batch in `data`.
"""
batchsize(A::BatchView) = A.size
numobs(A::BatchView) = A.count
Base.parent(A::BatchView) = A.data
Base.length(A::BatchView) = A.count
Base.getindex(A::BatchView, batchindex::Int) =
    datasubset(A.data, _batchrange(A, batchindex))

function Base.getindex(A::BatchView, batchindices::AbstractVector)
    obsindices = union((_batchrange(A, bi) for bi in batchindices)...)::Vector{Int}
    BatchView(datasubset(A.data, obsindices); A.size, A.partial)
end


# Helper function to translate a batch-index into a range of observations.
function _batchrange(a::BatchView, batchindex::Int)
    (batchindex > a.count || batchindex < 0) && throw(BoundsError())
    startidx = (batchindex - 1) * a.size + 1
    endidx = min(numobs(a.data), startidx + a.size -1) 
    return startidx:endidx
end


@doc (@doc BatchView)
const batchview = BatchView

datasubset(A::BatchView, i) = A[i]

function Base.showarg(io::IO, A::BatchView, toplevel)
    print(io, "batchview(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, "size=$(A.size), ")
    print(io, "partial=$(A.partial)")
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simplify
end

# --------------------------------------------------------------------

