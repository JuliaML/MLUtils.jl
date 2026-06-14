# """
#     eachobsparallel(data; buffer, nworkers, channelsize)

# Construct a data iterator over observations in container `data`.
# It uses worker tasks spread over the available threads to load
# observations in parallel, leading to large speedups when threads
# are available.

# To ensure that the active Julia session has multiple threads
# available, check that `Threads.nthreads() > 1`. You can start
# Julia with multiple threads with the `-t n` option. If your data
# loading is bottlenecked by the CPU, it is recommended to set `n`
# to the number of physical CPU cores.

# ## Arguments

# - `data`: a data container that implements `getindex/getobs` and `length/numobs`
# - `buffer = false`: whether to use inplace data loading with `getobs!`. Only use
#     this if you need the additional performance and `getobs!` is implemented for
#     `data`. Setting `buffer = true` means that when using the iterator, an
#     observation is only valid for the current loop iteration.
#     You can also pass in a preallocated `buffer = getobs(data, 1)`.
# - `nworkers = Threads.nthreads()`: the number of worker tasks that load observations
#     concurrently. Each in-flight worker holds one (possibly batched) observation, so
#     lowering `nworkers` is the main lever for capping peak memory.
# - `channelsize = nworkers`: the number of observations that are prefetched.
#     Increasing `channelsize` can lead to speedups when per-observation processing
#     time is irregular but will cause higher memory usage.
# """
function eachobsparallel(
        data;
        buffer::Bool = false,
        nworkers::Int = Threads.nthreads(),
        channelsize::Int = nworkers)
    if buffer
        return _eachobsparallel_buffered(buffer, data; nworkers, channelsize)
    else
        return _eachobsparallel_unbuffered(data; nworkers, channelsize)
    end
end

function _eachobsparallel_buffered(
        buffer,
        data;
        nworkers::Int,
        channelsize::Int = nworkers)
    buffers = [buffer]
    foreach(_ -> push!(buffers, deepcopy(buffer)), 1:channelsize)

    # `getobs!(buffer, data, i)` may return a value of a different type than
    # `buffer` itself: with `collate=true` it mutates the per-observation
    # buffers but returns a freshly batched array. Determine that result type
    # up front so the `RingBuffer`'s results channel stays concretely typed
    # (see JuliaML/MLUtils.jl#216).
    R = _buffered_result_type(data, buffer)

    # This ensures the `Loader` will take from the `RingBuffer`s result
    # channel, and that a new results channel is created on repeated
    # iteration. (Since `Loader`) closes the previous at the end of
    # each iteration.
    setup_channel(sz) = RingBuffer(buffers, R)

    return Loader(1:numobs(data); nworkers, channelsize, setup_channel) do ringbuffer, i
        # Internally, `RingBuffer` will `put!` the result in the results channel
        put!(ringbuffer) do buf
            getobs!(buf, data, i)
        end
    end
end

# Type of a collated batch (which `getobs!` returns) is recorded in `BatchView`'s
# first type parameter; for any other container the result keeps the buffer's type.
_buffered_result_type(data::BatchView, buffer) = eltype(data)
_buffered_result_type(data, buffer) = typeof(buffer)

function _eachobsparallel_unbuffered(data;
        nworkers::Int,
        channelsize::Int = nworkers
    )
    return Loader(1:numobs(data); nworkers, channelsize) do ch, i
        obs = getobs(data, i)
        put!(ch, obs)
    end
end


# ## Internals

# The `Loader` handles the asynchronous iteration and fills
# a result channel.


# """
#     Loader(f, args; nworkers, channelsize, setup_channel)

# Create a threaded iterator that iterates over `(f(arg) for arg in args)`
# using `nworkers` worker tasks that prefill a channel of length `channelsize`.

# Note: results may not be returned in the correct order.
# """
struct Loader
    f
    argiter::AbstractVector
    nworkers::Int
    channelsize::Int
    setup_channel
end

function Loader(
        f,
        argiter;
        nworkers::Int = Threads.nthreads(),
        channelsize::Int = nworkers,
        setup_channel = sz -> Channel(sz))
    Loader(f, argiter, nworkers, channelsize, setup_channel)
end

Base.length(loader::Loader) = length(loader.argiter)

struct LoaderState
    spawnertask::Any
    channel::Any
    remaining::Any
end

function Base.iterate(loader::Loader)
    ch = loader.setup_channel(loader.channelsize)
    basesize = length(loader.argiter) ÷ max(loader.nworkers, 1)
    task = Threads.@spawn begin
        try
            _spawn_foreach(loader.f, ch, loader.argiter,
                           firstindex(loader.argiter),
                           lastindex(loader.argiter),
                           basesize)
        catch e
            close(ch, e)
            rethrow()
        end
    end

    return Base.iterate(loader, LoaderState(task, ch, length(loader.argiter)))
end

# Recursive divide-and-conquer over `argiter[lo:hi]`:
# At each level we `@spawn` the right half and recurse on the left half on the current task, then `wait` on the right.
# Leaves of size `<= basesize` are processed sequentially.
function _spawn_foreach(f::F, ch, argiter, lo, hi, basesize::Int) where {F}
    if hi - lo < max(basesize, 1)
        for i in lo:hi
            f(ch, argiter[i])
        end
    else
        mid = (lo + hi) >> 1
        task = Threads.@spawn _spawn_foreach($f, $ch, $argiter, $(mid + 1), $hi, $basesize)
        # Always `wait` on the right half, even if the left half throws, so the spawned
        # task never outlives this call. On the happy path `wait` also propagates a
        # right-half failure; on the error path the caller closes `ch`, which unblocks any
        # `put!` the right half is parked on so it can terminate.
        try
            _spawn_foreach(f, ch, argiter, lo, mid, basesize)
        finally
            wait(task)
        end
    end
    return nothing
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

# """
#     RingBuffer(size, buffer)
#     RingBuffer(buffers)

# A `Channel`-like data structure that rotates through
# `size` buffers. You can either pass in a vector of `buffers`, or
# a single `buffer` that is copied `size` times.

# `put!`s work by mutating one of the buffers:

# ```
# put!(ringbuffer) do buf
#     rand!(buf)
# end
# ```

# The result can then be `take!`n:

# ```
# res = take!(ringbuffer)
# ```

# !!! warning "Invalidation"

#     Only one result is valid at a time! On the next `take!`, the previous
#     result will be reused as a buffer and be mutated by a `put!`
# """
# The results channel carries `(buffer, result)` pairs rather than bare results.
# This way the buffer that produced a result is recycled back into the pool, while
# the result handed to the consumer may have a different type `R` than the buffer
# type `B`. That happens with `collate=true`, where `f!` mutates per-observation
# buffers but returns a freshly batched array (see JuliaML/MLUtils.jl#216). Both
# types are tracked so the channels stay concretely typed.
mutable struct RingBuffer{B,R}
    buffers::Channel{B}
    results::Channel{Tuple{B,R}}
    current::B
end

# Single-argument form: the result of `f!` has the same type as the buffer
# (e.g. in-place `getobs!` without collation).
RingBuffer(bufs::Vector{B}) where {B} = RingBuffer(bufs, B)

function RingBuffer(bufs::Vector{B}, ::Type{R}) where {B,R}
    size = length(bufs) - 1
    ch_buffers = Channel{B}(size + 1)
    ch_results = Channel{Tuple{B,R}}(size)
    foreach(bufs[begin+1:end]) do buf
        put!(ch_buffers, buf)
    end

    return RingBuffer{B,R}(ch_buffers, ch_results, bufs[begin])
end


function Base.take!(ringbuffer::RingBuffer)
    # Recycle the buffer that produced the previously returned result. This is
    # deferred until now so a result aliasing its buffer (e.g. `collate=false`)
    # stays valid until the consumer asks for the next one.
    put!(ringbuffer.buffers, ringbuffer.current)
    buf, result = take!(ringbuffer.results)
    ringbuffer.current = buf
    return result
end


# """
#     put!(f!, ringbuffer::RingBuffer)

# Apply f! to a buffer in `ringbuffer` and put into the results
# channel.

# ```julia
# x = rand(10, 10)
# ringbuffer = RingBuffer(1, x)
# put!(ringbuffer) do buf
#     @test x == buf
#     copy!(buf, rand(10, 10))
# end
# x_ = take!(ringbuffer)
# @test !(x ≈ x_)

# ```
# """
function Base.put!(f!, ringbuffer::RingBuffer)
    buf = take!(ringbuffer.buffers)
    buf_ = f!(buf)
    put!(ringbuffer.results, (buf, buf_))
    return buf_
end

Base.put!(c::Channel, b::MLUtils.RingBuffer) = throw(MethodError(put!, (c, b)))

function Base.close(ringbuffer::RingBuffer, args...)
    close(ringbuffer.results, args...)
    close(ringbuffer.buffers, args...)
end
