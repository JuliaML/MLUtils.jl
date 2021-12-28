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


"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1, upto = false)
    num_observations = numobs(source)
    @assert num_observations > 0
    if upto && size > num_observations
        size = num_observations
    end
    size  <= num_observations || throw(ArgumentError("Specified batch-size is too large for the given number of observations"))
    count <= num_observations || throw(ArgumentError("Specified batch-count is too large for the given number of observations"))
    if size > 0 && upto
        while num_observations % size != 0 && size > 1
            size = size - 1
        end
    end
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = 1
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. try use all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # use count just for boundscheck
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(ArgumentError("Specified number of partitions is too large for the specified size"))
        count = max_batchcount
    end

    # check if the settings will result in all data points being used
    unused = num_observations - size*count
    if unused > 0
        @warn "The specified values for size and/or count will result in $unused unused data points" maxlog=1
    end
    size::Int, count::Int
end

"""
Helper function to translate a batch-index into a range of observations.
"""
function _batchrange(batchsize::Int, batchindex::Int)
    startidx = (batchindex - 1) * batchsize + 1
    startidx:(startidx + batchsize - 1)
end

# --------------------------------------------------------------------

"""
    BatchView(data, [size|maxsize], [count])

Description
============

Create a view of the given `data` that represents it as a vector
of batches. Each batch will contain an equal amount of
observations in them. The number of batches and the batch-size
can be specified using (keyword) parameters `count` and `size`.
In the case that the size of the dataset is not dividable by the
specified (or inferred) `size`, the remaining observations will
be ignored with a warning.

Note that any data access is delayed until `getindex` is called,
and even `getindex` returns the result of [`datasubset`](@ref)
which in general avoids data movement until [`getobs`](@ref) is
called.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
mini-batch of constant [`numobs`](@ref), which effectively allows
to iterator over [`data`](@ref) one batch at a time.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).

- **`size`** : The batch-size of each batch. I.e. the number of
    observations that each batch must contain.

- **`maxsize`** : The maximum batch-size of each batch. I.e. the
    number of observations that each batch should contain. If the
    number of total observation is not divideable by the size it
    will be reduced until it is.

- **`count`** : The number of batches that the view will contain.


Methods
========

Aside from the `AbstractArray` interface following additional
methods are provided.

- **`getobs(data::BatchView, batchindices)`** :
    Returns a `Vector` of the batches specified by `batchindices`.

- **`numobs(data::BatchView)`** :
    Returns the total number of observations in `data`. Note that
    unless the batch-size is 1, this number will differ from
    `length`.

Details
========

For `BatchView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See `?DataSubset` for more info.


Examples
=========

```julia
using MLDataUtils
X, Y = MLDataUtils.load_iris()

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
for (x,y) in batchview((X,Y), size = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert numobs(x) === numobs(y) === 20
end

# 10 batches of size 15 observations
for (x,y) in batchview((X,Y), maxsize = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert numobs(x) === numobs(y) === 15
end

# randomly assign observations to one and only one batch.
for (x,y) in batchview(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end

# iterate over the first 2 batches of 15 observation each
for (x,y) in batchview((X,Y), size=15, count=2)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert size(x) == (4, 15)
    @assert size(y) == (15,)
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
end

function BatchView(data::T, size::Int, count::Int, upto::Bool = false) where {T}
    nsize, ncount = _compute_batch_settings(data, size, count, upto)
    E = typeof(datasubset(data, 1:nsize))
    BatchView{E,T}(data, nsize, ncount)
end

function BatchView(data::T, size::Int, upto::Bool = false) where {T}
    BatchView(data, size, -1, upto)
end

function BatchView(A::BatchView, size::Int, count::Int, upto::Bool = false)
    BatchView(parent(A), size, count, upto)
end

BatchView(data) = BatchView(data, -1, -1)

function BatchView(data; size = -1, maxsize = -1, count = -1)
    maxsize != -1 && size != -1 && throw(ArgumentError("Providing both \"size\" and \"maxsize\" is not supported"))
    if maxsize != -1
        # set upto to true in order to allow a flexible batch size
        BatchView(data, maxsize, count, true)
    else
        # force given batch size
        BatchView(data, size, count)
    end
end

"""
    batchsize(data) -> Int

Return the fixed size of each batch in `data`.
"""
batchsize(A::BatchView) = A.size
numobs(A::BatchView) = A.count * A.size
Base.parent(A::BatchView) = A.data
Base.length(A::BatchView) = A.count
Base.getindex(A::BatchView, batchindex::Int) =
    datasubset(A.data, _batchrange(A.size, batchindex))

function Base.getindex(A::BatchView, batchindices::AbstractVector)
    obsindices = union((_batchrange(A.size, bi) for bi in batchindices)...)::Vector{Int}
    BatchView(datasubset(A.data, obsindices), A.size, -1)
end

@doc (@doc BatchView)
const batchview = BatchView

function Base.showarg(io::IO, A::BatchView, toplevel)
    print(io, "batchview(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, A.size, ", ")
    print(io, A.count)
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simpify
end

# --------------------------------------------------------------------

DataSubset(A::ObsView, i) = ObsView(DataSubset(parent(A), i))
datasubset(A::ObsView, i) = ObsView(datasubset(parent(A), i))
DataSubset(A::BatchView, i) = BatchView(DataSubset(parent(A), i), A.size, -1)
datasubset(A::BatchView, i) = BatchView(datasubset(parent(A), i), A.size, -1)
