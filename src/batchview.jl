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

A = BatchView(X, batchsize=30)
@assert typeof(A) <: BatchView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,2}
@assert length(A) == 5 # Iris has 150 observations
@assert size(A[1]) == (4,30) # Iris has 4 features

# 5 batches of size 30 observations
for x in BatchView(X, batchsize=30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert numobs(x) === 30
end

# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x, y) in BatchView((X, Y), batchsize=20, partial=false)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert numobs(x) == numobs(y) == 20
end


# randomly assign observations to one and only one batch.
for (x, y) in BatchView(shuffleobs((X, Y)), batchsize=20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end
```
"""
struct BatchView{TElem,TData,TCollate} <: AbstractDataContainer
    data::TData
    batchsize::Int
    count::Int
    partial::Bool
    imax::Int
end


function BatchView(data::T; batchsize::Int=1, partial::Bool=true, collate=Val(nothing)) where {T}
    n = numobs(data)
    if n < batchsize
        @warn "Number of observations less than batch-size, decreasing the batch-size to $n"
        batchsize = n
    end
    collate = collate isa Val ? collate : Val(collate)
    if !(collate ∈ (Val(nothing), Val(true), Val(false)))
        throw(ArgumentError("`collate` must be one of `nothing`, `true` or `false`."))
    end
    E = _batchviewelemtype(data, collate)
    imax = partial ? n : n - batchsize + 1
    count = partial ? ceil(Int, n / batchsize) : floor(Int, n / batchsize)
    BatchView{E,T,typeof(collate)}(data, batchsize, count, partial, imax)
end

_batchviewelemtype(::TData, ::Val{nothing}) where TData =
    Core.Compiler.return_type(getobs, Tuple{TData, UnitRange{Int}})
_batchviewelemtype(::TData, ::Val{false}) where TData =
    Vector{Core.Compiler.return_type(getobs, Tuple{TData, Int})}
_batchviewelemtype(data, ::Val{true}) =
    Core.Compiler.return_type(batch, Tuple{_batchviewelemtype(data, Val(false))})

"""
    batchsize(data) -> Int

Return the fixed size of each batch in `data`.
"""
batchsize(A::BatchView) = A.batchsize

Base.length(A::BatchView) = A.count

getobs(A::BatchView) = getobs(A.data)

function Base.getindex(A::BatchView, i::Int)
    obsindices = _batchrange(A, i)
    _getbatch(A, obsindices)
end

function Base.getindex(A::BatchView, is::AbstractVector)
    obsindices = union((_batchrange(A, i) for i in is)...)::Vector{Int}
    _getbatch(A, obsindices)
end

function _getbatch(A::BatchView{TElem, TData, Val{true}}, obsindices) where {TElem, TData}
    batch([getobs(A.data, i) for i in obsindices])
end
function _getbatch(A::BatchView{TElem, TData, Val{false}}, obsindices) where {TElem, TData}
    return [getobs(A.data, i) for i in obsindices]
end
function _getbatch(A::BatchView{TElem, TData, Val{nothing}}, obsindices) where {TElem, TData}
    getobs(A.data, obsindices)
end

Base.parent(A::BatchView) = A.data
Base.eltype(::BatchView{Tel}) where Tel = Tel

# override AbstractDataContainer default
Base.iterate(A::BatchView, state = 1) =
    (state > numobs(A)) ? nothing : (A[state], state + 1)

obsview(A::BatchView) = A
obsview(A::BatchView, i) = A[i]

# Helper function to translate a batch-index into a range of observations.
function _batchrange(a::BatchView, batchindex::Int)
    (batchindex > a.count || batchindex < 0) && throw(BoundsError())
    startidx = (batchindex - 1) * a.batchsize + 1
    endidx = min(numobs(a.data), startidx + a.batchsize -1)
    return startidx:endidx
end

function Base.showarg(io::IO, A::BatchView, toplevel)
    print(io, "BatchView(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, "batchsize=$(A.batchsize), ")
    print(io, "partial=$(A.partial)")
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simplify
end

# --------------------------------------------------------------------
