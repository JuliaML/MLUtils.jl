"""
    BatchView(data, batchsize; partial=true)
    BatchView(data; batchsize=1, partial=true)

Create a view of the given `data` that represents it as a vector
of batches. Each batch will contain an equal amount of
observations in them. The batch-size
can be specified using the  parameter `batchsize`.
In the case that the size of the dataset is not dividable by the
specified `batchsize`, the remaining observations will
be ignored if `partial=false`. If  `partial=true` instead 
the last batch-size can be slightly smaller.

Note that any data access is delayed until `getindex` is called,
and even `getindex` returns an [`ObsView`](@ref)
which in general avoids data movement until [`getobs`](@ref) is
called.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. 

For `BatchView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See [`ObsView`](@ref) for more info.

# Arguments

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`numobs`](@ref) (see Details for more information).

- **`batchsize`** : The batch-size of each batch. 
    It is the number of observations that each batch must contain
    (except possibly for the last one).

- **`partial`** : If `partial=false` and the number of observations is 
    not divisible by the batch-size, then the last mini-batch is dropped.

# Examples

```julia
using MLUtils
X, Y = MLUtils.load_iris()

A = batchview(X, batchsize = 30)
@assert typeof(A) <: BatchView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,2}
@assert length(A) == 5 # Iris has 150 observations
@assert size(A[1]) == (4,30) # Iris has 4 features

# 5 batches of size 30 observations
for x in batchview(X, batchsize = 30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert numobs(x) === 30
end

# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x,y) in batchview((X,Y), batchsize = 20, partial=false)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert numobs(x) == numobs(y) == 20
end


# randomly assign observations to one and only one batch.
for (x,y) in batchview(shuffleobs((X,Y)), batchsize=20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end
```
"""
struct BatchView{TElem,TData} <: AbstractDataContainer
    data::TData
    batchsize::Int
    count::Int
    partial::Bool
    imax::Int
end

@doc (@doc BatchView)
const batchview = BatchView

function BatchView(data::T; batchsize::Int=1, partial::Bool=true) where {T}
    n = numobs(data)
    if n < batchsize
        @warn "Number of observations less than batch-size, decreasing the batch-size to $n"
        batchsize = n
    end
    E = typeof(datasubset(data, 1:batchsize))
    imax = partial ? n : n - batchsize + 1
    count = partial ? ceil(Int, n / batchsize) : floor(Int, n / batchsize)
    BatchView{E,T}(data, batchsize, count, partial, imax)
end

"""
    batchsize(data) -> Int

Return the fixed size of each batch in `data`.
"""
batchsize(A::BatchView) = A.batchsize

numobs(A::BatchView) = A.count
getobs(A::BatchView) = getobs(A.data)
getobs(A::BatchView, i::Int) = getobs(A.data, _batchrange(A, i))

function getobs(A::BatchView, is::AbstractVector)
    obsindices = union((_batchrange(A, i) for i in is)...)::Vector{Int}
    getobs(A.data, obsindices)
end

Base.parent(A::BatchView) = A.data
Base.eltype(::BatchView{Tel}) where Tel = Tel

Base.getindex(A::BatchView, i::Int) = datasubset(A.data, _batchrange(A, i))

function Base.getindex(A::BatchView, is::AbstractVector)
    obsindices = union((_batchrange(A, i) for i in is)...)::Vector{Int}
    datasubset(A.data, obsindices)
end

datasubset(A::BatchView) = A
datasubset(A::BatchView, i) = A[i]

# Helper function to translate a batch-index into a range of observations.
function _batchrange(a::BatchView, batchindex::Int)
    (batchindex > a.count || batchindex < 0) && throw(BoundsError())
    startidx = (batchindex - 1) * a.batchsize + 1
    endidx = min(numobs(a.data), startidx + a.batchsize -1) 
    return startidx:endidx
end

function Base.showarg(io::IO, A::BatchView, toplevel)
    print(io, "batchview(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, "batchsize=$(A.batchsize), ")
    print(io, "partial=$(A.partial)")
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simplify
end

# --------------------------------------------------------------------

