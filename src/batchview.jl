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
and even `getindex` returns the result of [`obsview`](@ref)
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
interface. See `?ObsView` for more info.


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
[`getobs`](@ref), [`numobs`](@ref), [`ObsView`](@ref)
"""
struct BatchView{TElem,TData} <: AbstractDataContainer
    data::TData
    size::Int
    count::Int
    partial::Bool
    imax::Int
end

@doc (@doc BatchView)
const batchview = BatchView

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
getobs(A::BatchView) = getobs(A.data)
getobs(A::BatchView, i) = getobs(A[i])

Base.parent(A::BatchView) = A.data
Base.eltype(::BatchView{Tel}) where Tel = Tel

Base.getindex(A::BatchView, batchindex::Int) =
    datasubset(A.data, _batchrange(A, batchindex))

function Base.getindex(A::BatchView, batchindices::AbstractVector)
    obsindices = union((_batchrange(A, bi) for bi in batchindices)...)::Vector{Int}
    datasubset(A.data, obsindices)
end

datasubset(A::BatchView) = A
datasubset(A::BatchView, i) = A[i]

# Helper function to translate a batch-index into a range of observations.
function _batchrange(a::BatchView, batchindex::Int)
    (batchindex > a.count || batchindex < 0) && throw(BoundsError())
    startidx = (batchindex - 1) * a.size + 1
    endidx = min(numobs(a.data), startidx + a.size -1) 
    return startidx:endidx
end

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

