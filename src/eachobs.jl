"""
    eachobs(data, buffer=false, batchsize=-1, partial=true)

Return an iterator over the observations in `data`.

# Arguments

- `data`. The data to be iterated over. The data type has to implement
  [`numobs`](@ref) and [`getobs`](@ref).
- `buffer`. If `buffer=true` and supported by the type of `data`,
a buffer will be allocated and reused for memory efficiency.
You can also pass a preallocated object to `buffer`.
- `batchsize`. If less than 0, iterates over individual observation.
Otherwise, each iteration (except possibly the last) yields a mini-batch
containing `batchsize` observations.
- `partial`. This argument is used only when `batchsize > 0`.
  If `partial=false` and the number of observations is not divisible by the batchsize,
  then the last mini-batch is dropped.
- `parallel=false`. Whether to use load data in parallel using worker threads. Greatly
    speeds up data loading by factor of available threads. Requires starting
    Julia with multiple threads. Check `Threads.nthreads()` to see the number of
    available threads. **Passing `parallel = true` breaks ordering guarantees**
- `shuffle = false`: Whether to shuffle the observations before iterating. Unlike
    wrapping the data container with `shuffleobs(data)`, `shuffle = true` ensures
    that the observations are shuffled anew every time you start iterating over
    `eachobs`.

See also [`numobs`](@ref), [`getobs`](@ref).

# Examples

```julia
X = rand(4,100)
for x in eachobs(X)
    # loop entered 100 times
    @assert typeof(x) <: Vector{Float64}
    @assert size(x) == (4,)
end

# mini-batch iterations
for x in eachobs(X, batchsize=10)
    # loop entered 10 times
    @assert typeof(x) <: Matrix{Float64}
    @assert size(x) == (4,10)
end

# support for tuples, named tuples, dicts
for (x, y) in eachobs((X, Y))
    # ...
end
```
"""
function eachobs(
        data;
        buffer = false,
        parallel = false,
        shuffle = false,
        batchsize::Int = -1,
        partial::Bool = true,
        executor = _default_executor())
    if batchsize > 0
        data = BatchView(data; batchsize, partial)
    end

    iter = if parallel
        eachobsparallel(data; buffer, executor)
    else
        if buffer === false
            EachObs(data)
        elseif buffer === true
            EachObsBuffer(data)
        else
            EachObsBuffer(data, buffer)
        end
    end

    if shuffle
        iter = ReshuffleIter(iter)
    end
    return iter
end


# Internal

"""
    EachObs(data)

Create an iterator over observations in data container `data`.

This is an internal function. Use `eachobs(data)` instead.
"""
struct EachObs{T}
    data::T
end
Base.length(iter::EachObs) = numobs(iter.data)
Base.eltype(iter::EachObs) = eltype(iter.data)

function Base.iterate(iter::EachObs, i::Int = 1)
    i > numobs(iter) && return nothing
    return getobs(iter.data, i), i+1
end


"""
    EachObsBuffer(data)

Create a buffered iterator over observations in data container `data`.
Buffering only works if [`getobs!`](@ref) is implemented for `data`.

This is an internal function. Use `eachobs(data, buffer = true)` instead.
"""
struct EachObsBuffer{T, B}
    data::T
    buffer::B
end
EachObsBuffer(data) = EachObsBuffer(data, getobs(data, 1))
Base.length(iter::EachObsBuffer) = numobs(iter.data)
Base.eltype(iter::EachObsBuffer) = eltype(iter.data)

function Base.iterate(iter::EachObsBuffer)
    obs = getobs!(iter.buffer, iter.data, 1)
    return obs, (obs, 2)
end

function Base.iterate(iter::EachObsBuffer, (buffer, i))
    i > numobs(iter) && return nothing
    return getobs!(buffer, iter.data, i), (buffer, i+1)
end
