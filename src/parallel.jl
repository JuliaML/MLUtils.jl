"""
    eachobsparallel(data; buffer, executor, channelsize)

Construct a data iterator over observations in container `data`.
It uses available threads as workers to load observations in
parallel, leading to large speedups when threads are available.

To ensure that the active Julia session has multiple threads
available, check that `Threads.nthreads() > 1`. You can start
Julia with multiple threads with the `-t n` option. If your data
loading is bottlenecked by the CPU, it is recommended to set `n`
to the number of physical CPU cores.

## Arguments

- `data`: a data container that implements `getindex/getobs` and `length/numobs`
- `buffer = false`: whether to use inplace data loading with `getobs!`. Only use
    this if you need the additional performance and `getobs!` is implemented for
    `data`. Setting `buffer = true` means that when using the iterator, an
    observation is only valid for the current loop iteration.
    You can also pass in a preallocated `buffer = getobs(data, 1)`.
- `executor = Folds.ThreadedEx()`: task scheduler
    You may specify a different task scheduler which can
    be any `Folds.Executor`.
- `channelsize = Threads.nthreads()`: the number of observations that are prefetched.
    Increasing `channelsize` can lead to speedups when per-observation processing
    time is irregular but will cause higher memory usage.
"""
function eachobsparallel(
        data;
        executor::Executor = _default_executor(),
        buffer = false,
        channelsize = Threads.nthreads())
    if buffer
        return _eachobsparallel_buffered(data, executor; channelsize)
    else
        return _eachobsparallel_unbuffered(data, executor; channelsize)
    end
end


function _eachobsparallel_buffered(
        data,
        executor;
        buffer = getobs(data, 1),
        channelsize=Threads.nthreads())
    buffers = [buffer]
    foreach(_ -> push!(buffers, deepcopy(buffer)), 1:channelsize)

    # This ensures the `Loader` will take from the `RingBuffer`s result
    # channel, and that a new results channel is created on repeated
    # iteration. (Since `Loader`) closes the previous at the end of
    # each iteration.
    setup_channel(sz) = RingBuffer(buffers)

    return Loader(1:numobs(data); executor, channelsize, setup_channel) do ringbuffer, i
        # Internally, `RingBuffer` will `put!` the result in the results channel
        put!(ringbuffer) do buf
            getobs!(buf, data, i)
        end
    end
end

function _eachobsparallel_unbuffered(data, executor; channelsize=Threads.nthreads())
    return Loader(1:numobs(data); executor, channelsize) do ch, i
        obs = getobs(data, i)
        put!(ch, obs)
    end
end


# Unlike DataLoaders.jl, this currently does not use task pools
# since  `ThreadedEx` has shown to be more performant. This may
# change in the future.
# See PR 33 https://github.com/JuliaML/MLUtils.jl/pull/33
_default_executor() = ThreadedEx()


# ## Internals

# The `Loader` handles the asynchronous iteration and fills
# a result channel.


"""
    Loader(f, args; executor, channelsize, setup_channel)

Create a threaded iterator that iterates over `(f(arg) for arg in args)`
using threads that prefill a channel of length `channelsize`.

Note: results may not be returned in the correct order, depending on
`executor`.
"""
struct Loader
    f
    argiter::AbstractVector
    executor::Executor
    channelsize::Int
    setup_channel
end

function Loader(
        f,
        argiter;
        executor=_default_executor(),
        channelsize=Threads.nthreads(),
        setup_channel = sz -> Channel(sz))
    Loader(f, argiter, executor, channelsize, setup_channel)
end

Base.length(loader::Loader) = length(loader.argiter)

struct LoaderState
    spawnertask::Any
    channel::Any
    remaining::Any
end

function Base.iterate(loader::Loader)
    ch = loader.setup_channel(loader.channelsize)
    task = @async begin
        @floop loader.executor for arg in loader.argiter
            try
                loader.f(ch, arg)
            catch e
                close(ch, e)
                rethrow()
            end
        end
    end

    return Base.iterate(loader, LoaderState(task, ch, length(loader.argiter)))
end

function Base.iterate(::Loader, state::LoaderState)
    if state.remaining == 0
        close(state.channel)
        return nothing
    else
        result = take!(state.channel)
        return result, LoaderState(state.spawnertask, state.channel, state.remaining - 1)
    end
end


# The `RingBuffer` ensures that the same buffers are reused
# and the loading works asynchronously.

"""
    RingBuffer(size, buffer)
    RingBuffer(buffers)

A `Channel`-like data structure that rotates through
`size` buffers. You can either pass in a vector of `buffers`, or
a single `buffer` that is copied `size` times.

`put!`s work by mutating one of the buffers:

```
put!(ringbuffer) do buf
    rand!(buf)
end
```

The result can then be `take!`n:

```
res = take!(ringbuffer)
```

!!! warning "Invalidation"

    Only one result is valid at a time! On the next `take!`, the previous
    result will be reused as a buffer and be mutated by a `put!`
"""
mutable struct RingBuffer{T}
    buffers::Channel{T}
    results::Channel{T}
    current::T
end

function RingBuffer(bufs::Vector{T}) where T
    size = length(bufs) - 1
    ch_buffers = Channel{T}(size + 1)
    ch_results = Channel{T}(size)
    foreach(bufs[begin+1:end]) do buf
        put!(ch_buffers, buf)
    end

    return RingBuffer{T}(ch_buffers, ch_results, bufs[begin])
end


function Base.take!(ringbuffer::RingBuffer)
    put!(ringbuffer.buffers, ringbuffer.current)
    ringbuffer.current = take!(ringbuffer.results)
    return ringbuffer.current
end


"""
    put!(f!, ringbuffer::RingBuffer)

Apply f! to a buffer in `ringbuffer` and put into the results
channel.

```julia
x = rand(10, 10)
ringbuffer = RingBuffer(1, x)
put!(ringbuffer) do buf
    @test x == buf
    copy!(buf, rand(10, 10))
end
x_ = take!(ringbuffer)
@test !(x ≈ x_)

```
"""
function Base.put!(f!, ringbuffer::RingBuffer)
    buf = take!(ringbuffer.buffers)
    buf_ = f!(buf)
    put!(ringbuffer.results, buf_)
    return buf_
end

Base.put!(c::Channel{T}, b::MLUtils.RingBuffer) = throw(MethodError(put!, (c, b)))

function Base.close(ringbuffer::RingBuffer, args...)
    close(ringbuffer.results, args...)
    close(ringbuffer.buffers, args...)
end
