"""
    BatchView(data, batchsize; partial=true, collate=nothing)
    BatchView(data; batchsize=1, partial=true, collate=nothing)

Create a view of the given `data` that represents it as a vector
of batches. Each batch will contain an equal amount of
observations in them. The batch-size
can be specified using the  parameter `batchsize`.
In the case that the size of the dataset is not dividable by the
specified `batchsize`, the remaining observations will
be ignored if `partial=false`. If  `partial=true` instead
the last batch-size can be slightly smaller.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. 

Any data access is delayed until iteration or indexing is perfomed. 
The [`getobs`](@ref MLUtils.getobs) function is called on the data object to retrieve the
observations.

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

- **`collate`**: Defines the batching behavior. 
    - If `nothing` (default), a batch is `getobs(data, indices)`. 
    - If `false`, each batch is `[getobs(data, i) for i in indices]`. 
    - If `true`, applies MLUtils to the vector of observations in a batch, 
      recursively collating arrays in the last dimensions. See [`MLUtils.batch`](@ref) for more information
      and examples.
    - If a custom function, it will be used in place of `MLUtils.batch`. It should take a vector of observations as input.


Se also [`DataLoader`](@ref).

# Examples

```jldoctest
julia> using MLUtils

julia> X, Y = MLUtils.load_iris();

julia> A = BatchView(X, batchsize=30);

julia> @assert eltype(A) <: Matrix{Float64}

julia> @assert length(A) == 5 # Iris has 150 observations

julia> @assert size(A[1]) == (4,30) # Iris has 4 features

julia> for x in BatchView(X, batchsize=30)
           # 5 batches of size 30 observations
           @assert size(x) == (4, 30)
           @assert numobs(x) === 30
       end

julia> for (x, y) in BatchView((X, Y), batchsize=20, partial=true)
           # 7 batches of size 20 observations + 1 batch of 10 observations
           @assert typeof(x) <: Matrix{Float64}
           @assert typeof(y) <: Vector{String}
       end

julia> for batch in BatchView((X, Y), batchsize=20, partial=false, collate=false)
           # 7 batches of size 20 observations
           @assert length(batch) == 20
           x1, y1 = batch[1]
       end

julia> function collate_fn(batch)
           # collate observations into a custom batch
           return hcat([x[1] for x in batch]...), join([x[2] for x in batch])
        end;

julia> for (x, y) in BatchView((rand(10, 4), ["a", "b", "c", "d"]), batchsize=2, collate=collate_fn)
           @assert size(x) == (10, 2)
           @assert y isa String
       end
```
"""
struct BatchView{TElem,TData,TCollate} <: AbstractDataContainer
    data::TData
    batchsize::Int
    count::Int
    partial::Bool
    collate::TCollate # either Val(nothing), Val(false), or a function
end

function BatchView(data::T; batchsize::Int=1, partial::Bool=true, collate=Val(nothing)) where {T}
    n = numobs(data)
    if n < batchsize
        @warn "Number of observations less than batch-size, decreasing the batch-size to $n"
        batchsize = n
    end
    if collate === nothing || collate isa Bool
        collate = Val(collate)
    end
    if collate === Val(true)
        collate = MLUtils.batch
    end
    E = _batchviewelemtype(data, collate)
    count = partial ? cld(n, batchsize) : fld(n, batchsize)
    return BatchView{E,T,typeof(collate)}(data, batchsize, count, partial, collate)
end

_batchviewelemtype(::TData, ::Val{nothing}) where TData =
    Core.Compiler.return_type(getobs, Tuple{TData, UnitRange{Int}})
_batchviewelemtype(::TData, ::Val{false}) where TData =
    Vector{Core.Compiler.return_type(getobs, Tuple{TData, Int})}
_batchviewelemtype(data, collate) =
    Core.Compiler.return_type(collate, Tuple{_batchviewelemtype(data, Val(false))})

function Base.show(io::IO, A::BatchView)
    print(io, "BatchView(")
    show(io, A.data)
    print(io, ", batchsize=$(A.batchsize), partial=$(A.partial), collate=$(A.collate)")
    print(io, ')')
end

"""
    batchsize(data::BatchView) -> Int

Return the fixed size of each batch in `data`.

# Examples

```julia
using MLUtils
X, Y = MLUtils.load_iris()

A = BatchView(X, batchsize=30)
@assert batchsize(A) == 30
```
"""
batchsize(A::BatchView) = A.batchsize

Base.length(A::BatchView) = A.count

Base.@propagate_inbounds function getobs(A::BatchView)
    return _getbatch(A, 1:numobs(A.data))
end

Base.@propagate_inbounds function Base.getindex(A::BatchView, i::Int)
    obsindices = _batchrange(A, i)
    return _getbatch(A, obsindices)
end

Base.@propagate_inbounds function Base.getindex(A::BatchView, is::AbstractVector)
    obsindices = union((_batchrange(A, i) for i in is)...)::Vector{Int}
    return _getbatch(A, obsindices)
end

function getobs!(buffer, A::BatchView, i::Int)
    obsindices = _batchrange(A, i)
    return _getbatch!(buffer, A, obsindices)
end

function _getbatch(A::BatchView{TElem,TData,TCollate}, obsindices) where {TElem,TData,TCollate}
    return A.collate([getobs(A.data, i) for i in obsindices])
end
function _getbatch!(buffer, A::BatchView{TElem,TData,TCollate}, obsindices) where {TElem,TData,TCollate}
    return A.collate([getobs!(buffer[i], A.data, i) for i in obsindices])
end

function _getbatch(A::BatchView{TElem,TData,Val{false}}, obsindices) where {TElem,TData}
    return [getobs(A.data, i) for i in obsindices]
end
function _getbatch!(buffer, A::BatchView{TElem,TData,Val{false}}, obsindices) where {TElem,TData}
    return [getobs!(buffer[i], A.data, i) for i in obsindices]
end

function _getbatch(A::BatchView{TElem,TData,Val{nothing}}, obsindices) where {TElem,TData}
    return getobs(A.data, obsindices)
end
function _getbatch!(buffer, A::BatchView{TElem,TData,Val{nothing}}, obsindices) where {TElem,TData,TCollate}
    return getobs!(buffer, A.data, obsindices)
end



Base.parent(A::BatchView) = A.data
Base.eltype(::BatchView{Tel}) where Tel = Tel

# override AbstractDataContainer default
Base.iterate(A::BatchView, state = 1) =
    (state > numobs(A)) ? nothing : (A[state], state + 1)

# Helper function to translate a batch-index into a range of observations.
@inline function _batchrange(A::BatchView, batchindex::Int)
    @boundscheck (batchindex > A.count || batchindex < 0) && throw(BoundsError())
    startidx = (batchindex - 1) * A.batchsize + 1
    endidx = min(numobs(parent(A)), startidx + A.batchsize -1)
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
