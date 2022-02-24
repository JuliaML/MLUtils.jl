"""
    eachobsparallel(data)

Construct a data iterator over observations in container `data`.
It uses available threads as workers to load observations in
parallel, leading to large speedups when threads are available.
"""
function eachobsparallel(
        data,
        executor::Executor;
        buffersize = Threads.nthreads())
    return Loader(executor, buffersize, 1:numobs(data)) do i
        getobs(data, i)
    end
end

# The default executor behaves similarly to DataLoaders.jl and allows
# loading data in the background
function eachobsparallel(data; background = true, kwargs...)
    return eachobsparallel(data, TaskPoolEx(background=background); kwargs...)
end


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
