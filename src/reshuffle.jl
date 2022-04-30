
"""
    ReshuffleIter(iter)

Wrap a data iterator `iter` created by [`eachobs`](@ref) so that it is
shuffled anew every time it is iterated over.

This is an internal function. Use `eachobs(data; shuffle = true)` instead.
"""
struct ReshuffleIter{T}
    iter::T
end
Base.length(iter::ReshuffleIter) = length(iter.iter)
Base.eltype(iter::ReshuffleIter) = eltype(iter.iter)

function Base.iterate(re::ReshuffleIter)
    iter = reshuffle(re.iter)
    el, state = iterate(iter)
    return el, (iter, state)
end
function Base.iterate(::ReshuffleIter, (iter, state))
    ret = iterate(iter, state)
    isnothing(ret) && return ret
    el, state = ret
    return el, (iter, state)
end

reshuffle(iter::EachObs) = EachObs(shuffleobs(iter.data))
reshuffle(iter::EachObsBuffer) = EachObsBuffer(shuffleobs(iter.data))
reshuffle(iter::MLUtils.Loader) = MLUtils.Loader(
    iter.f,
    collect(shuffleobs(iter.argiter)),  # shuffles the indices
    iter.executor,
    iter.channelsize,
    iter.setup_channel,
)
