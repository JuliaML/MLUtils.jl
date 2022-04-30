"""
    eachobs(data; [buffer, batchsize, partial, parallel, shuffle])

Return an iterator over the observations in `data`.

# Arguments

- `data`. The data to be iterated over. The data type has to be supported by
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
        rng::AbstractRNG = Random.GLOBAL_RNG)
    buffer = buffer isa Bool ? buffer : true
    return EachObs(data, batchsize, buffer, partial, shuffle, parallel, rng)
end

struct EachObs{T, R<:AbstractRNG}
    data::T
    batchsize::Int
    buffer::Bool
    partial::Bool
    shuffle::Bool
    parallel::Bool
    rng::R
end


function Base.iterate(iter::EachObs)
    data, shuffle, batchsize, partial, parallel, buffer, rng = iter.data,
        iter.shuffle, iter.batchsize, iter.partial, iter.parallel, iter.buffer,
        iter.rng

    data = e.shuffle ? shuffleobs(e.rng, e.data) : e.data
    data = e.batchsize > 0 ? BatchView(data; e.batchsize, e.partial) : data

    iter = if e.parallel
        eachobsparallel(data; e.buffer)
    else
        if e.buffer
            buf = getobs(data, 1)
            (getobs!(buf, data, i) for i in 1:numobs(data))
        else
            (getobs(data, i) for i in 1:numobs(data))
        end
    end
    obs, state = iterate(iter)
    return obs, (iter, state)
end


function Base.iterate(::EachObs, (iter, state))
    ret = iterate(iter, state)
    isnothing(ret) && return
    obs, state = ret
    return obs, (iter, state)
end


function Base.length(e::EachObs)
    numobs(if e.batchsize > 0
        BatchView(e.data; e.batchsize, e.partial)
    else
        e.data
    end)
end

function Base.eltype(e::EachObs)
    eltype(if e.batchsize > 0
        BatchView(e.data; e.batchsize, e.partial)
    else
        e.data
    end)
end
