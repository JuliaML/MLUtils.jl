"""
    eachobsparallel(data)
    eachobsparallel(data[, executor; buffersize])

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
- `executor = Folds.ThreadedEx()`: task scheduler

    You may specify a different task scheduler which can
    be any `Folds.Executor`.
- `buffersize = Threads.nthreads()`: the number of observations that are prefetched.

    Increasing `buffersize` can lead to speedups when per-observation processing
    time is irregular but will cause higher memory usage.
"""
function eachobsparallel(
        data,
        executor::Executor = _default_executor();
        buffersize = Threads.nthreads())
    return Loader(executor, buffersize, 1:numobs(data)) do i
        getobs(data, i)
    end
end

# Unlike DataLoaders.jl, this currently does not use task pools
# since  `ThreadedEx` has shown to be more performant. This may
# change in the future.
# See PR 33 https://github.com/JuliaML/MLUtils.jl/pull/33
_default_executor() = ThreadedEx(basesize=1)


mutable struct Loader
    f::Any
    executor::Any
    channelsize::Any
    argiter::Any
end

Base.length(loader::Loader) = length(loader.argiter)

struct LoaderState
    spawnertask::Any
    channel::Any
    remaining::Any
end

function Base.iterate(loader::Loader)
    ch = Channel(loader.channelsize)
    task = @async begin

        @floop loader.executor for arg in loader.argiter
            try
                result = loader.f(arg)
                @async put!(ch, result)
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
