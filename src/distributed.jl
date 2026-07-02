# Distributed (multiprocess) data loading for `DataLoader`.
#
# Opt-in via the `num_workers` keyword: `getobs`/`collate` run in separate worker
# *processes* (via `Distributed`) instead of threads. This unlocks parallel loading
# for containers whose `getobs` does not scale under threads — most importantly
# `PythonCall`-backed datasets, where a single shared CPython GIL serializes all reads.
#
# The container contract is unchanged (`getobs`/`numobs`); the only extra requirement is
# that the container — and the values returned by `getobs` — be serializable with Julia's
# stdlib `Serialization`, the analog of PyTorch requiring a *picklable* dataset for its
# spawned workers. Arrays, tuples, named tuples already satisfy this.

# ---------------------------------------------------------------------------
# Worker-side batch function
# ---------------------------------------------------------------------------

# The closure shipped to workers captures the (serializable) `data` container and the
# `collate` function, so that a `CachingPool` sends them exactly once per worker; after
# that only index sets travel out and collated Julia arrays travel back. It is built as
# an anonymous function (a `Function`, as `CachingPool` requires) by `_worker_closure`.
_worker_closure(collate, data) = idxs -> _worker_getbatch(collate, data, idxs)

# Mirror `BatchView`'s `_getbatch` collate semantics, but starting from the raw data
# container and a set of *original* observation indices computed on the main process.
_worker_getbatch(::Nothing, data, idx) = getobs(data, idx)                 # single obs (batchsize <= 0)
_worker_getbatch(::Val{nothing}, data, idxs) = getobs(data, idxs)          # collate=nothing
_worker_getbatch(::Val{false}, data, idxs) = [getobs(data, i) for i in idxs] # collate=false
_worker_getbatch(collate, data, idxs) = collate([getobs(data, i) for i in idxs]) # collate function (incl. Val(true) -> batch)

# ---------------------------------------------------------------------------
# Index sets (computed on the main process, honoring shuffle)
# ---------------------------------------------------------------------------

# Return `(idxsets, collate)` where `idxsets` is a vector of *original-data* observation
# index sets — one per batch — and `collate` is the resolved batching function.
function _distributed_index_sets(d::DataLoader)
    data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
    return _batch_index_sets(data)
end

# `BatchView` wraps an `ObsView`; translate each batch's positions into original indices.
function _batch_index_sets(bv::BatchView)
    ov = bv.data::ObsView
    idxsets = [ov.indices[_batchrange(bv, i)] for i in 1:bv.count]
    return idxsets, bv.collate
end

# `ObsView` directly (batchsize <= 0): each element is a single original index; no collate.
function _batch_index_sets(ov::ObsView)
    return collect(ov.indices), nothing
end

# ---------------------------------------------------------------------------
# Worker-pool management
# ---------------------------------------------------------------------------

# A persistent, lazily-grown set of managed worker processes (like PyTorch's
# `persistent_workers=True`), reused across loaders and epochs to amortize startup. Stored
# as an ordered pid vector so that slicing `[1:n]` is deterministic: a `num_workers=n`
# loader always binds to the *same* first `n` workers, and two loaders requesting the same
# `n` share them rather than fighting over the whole set. The lock makes concurrent
# first-iterations grow the set atomically (no double `addprocs`, no orphaned pids).
const _POOL_PIDS = Ref{Vector{Int}}(Int[])
const _POOL_LOCK = ReentrantLock()

# Ensure at least `n` managed worker processes exist; return the full ordered pid vector.
function _ensure_pool(n::Int, data)
    pids = lock(_POOL_LOCK) do
        pids = _POOL_PIDS[]
        if length(pids) < n
            # Match the parent's environment so workers instantiate the same project (and,
            # for HF datasets, the same CondaPkg Python). `active_project()` is usually a path.
            proj = Base.active_project()
            exeflags = proj === nothing ? `` : `--project=$proj`
            newpids = addprocs(n - length(pids); exeflags)
            pids = vcat(pids, newpids)
            _POOL_PIDS[] = pids
        end
        return pids
    end
    _load_modules_on_workers(data, pids)
    return pids
end

# Are all of `pids` still alive? (False after `close_dataloader_pool`, so a stale cached
# `CachingPool` gets rebuilt rather than dispatched onto dead workers.)
_pool_alive(pids) = !isempty(pids) && all(in(Distributed.procs()), pids)

# Make MLUtils (needed to deserialize the worker closure and call `getobs`) and the data
# container's defining package available on each worker so it can deserialize `data`.
function _load_modules_on_workers(data, pids)
    isempty(pids) && return nothing
    Distributed.remotecall_eval(Main, pids, :(import MLUtils))
    mod = parentmodule(typeof(data))
    if !(mod === Main || mod === Base || mod === Core || mod === MLUtils)
        name = nameof(mod)
        try
            Distributed.remotecall_eval(Main, pids, :(using $name))
        catch e
            error("MLUtils: could not auto-load module `$name` on worker processes for " *
                  "distributed data loading. Load it manually with `@everywhere using $name` " *
                  "before iterating the DataLoader. Original error: $e")
        end
    end
    return nothing
end

# Internal: terminate the persistent worker-process pool started by
# `DataLoader(...; num_workers=N)`. Workers are otherwise kept warm across loaders and
# epochs to amortize startup, and are child processes killed automatically on exit; this
# is only for releasing them earlier (e.g. to reclaim memory in a long-lived session).
function close_dataloader_pool()
    lock(_POOL_LOCK) do
        isempty(_POOL_PIDS[]) || rmprocs(_POOL_PIDS[]...)
        _POOL_PIDS[] = Int[]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

# A `Channel`-backed iterator over collated batches produced on worker processes.
# Mirrors the threaded `Loader` in `parallel.jl`, with `remotecall_fetch` over a
# `CachingPool` in place of `Threads.@spawn`.
mutable struct DistributedLoader
    channel::Channel
    task::Union{Task,Nothing}
    len::Int

    function DistributedLoader(channel::Channel, task, len::Int)
        dl = new(channel, task, len)
        # If the consumer stops early (`first`, `break`, an error in the loop), the
        # iterator state becomes unreferenced; closing the channel then unblocks a feeder
        # parked on `put!` so it unwinds instead of leaking a task and busy workers.
        finalizer(dl -> isopen(dl.channel) && close(dl.channel), dl)
        return dl
    end
end

Base.length(dl::DistributedLoader) = dl.len

function Base.iterate(dl::DistributedLoader, remaining::Int = dl.len)
    remaining == 0 && return nothing
    # `take!` rethrows if the feeder closed the channel with an exception.
    result = take!(dl.channel)
    return result, remaining - 1
end

function _distributed_loader(d::DataLoader)
    idxsets, collate = _distributed_index_sets(d)

    # Nothing to load: don't spin up worker processes at all.
    if isempty(idxsets)
        ch = Channel(0); close(ch)
        return DistributedLoader(ch, nothing, 0)
    end

    # Cache this loader's own `CachingPool` (over exactly its `num_workers` workers) + the
    # closure, so that `data` stays cached on the workers across epochs (only the shuffled
    # index sets change). Rebuild if those workers were torn down (e.g. by
    # `close_dataloader_pool`) — checked against the cached pids, not the shared global set.
    cache = d._cache[]
    if cache === nothing || !_pool_alive(cache.pids)
        allpids = _ensure_pool(d.num_workers, d.data)
        # Bind to exactly `num_workers` workers so distinct loaders don't grab the whole,
        # possibly larger, shared set (deterministic slice: same request → same workers).
        pids = allpids[1:min(d.num_workers, length(allpids))]
        cpool = CachingPool(pids)
        f = _worker_closure(collate, d.data)
        cache = (pids = pids, cpool = cpool, f = f)
        d._cache[] = cache
    end
    pids, cpool, f = cache.pids, cache.cpool, cache.f

    nworkers = max(length(pids), 1)
    ch = Channel(nworkers)   # bounded prefetch → overlap + backpressure
    task = Threads.@spawn begin
        try
            asyncmap(idxs -> put!(ch, remotecall_fetch(f, cpool, idxs)), idxsets; ntasks = nworkers)
            close(ch)
        catch e
            # A worker `getobs` failure is surfaced to the consumer's `take!`; but if the
            # channel is already closed (consumer abandoned iteration, see the finalizer
            # above), just unwind quietly.
            isopen(ch) ? close(ch, e) : nothing
        end
    end
    return DistributedLoader(ch, task, length(idxsets))
end

function Base.iterate(d::DataLoader{T,B,:distributed}) where {T,B}
    dl = _distributed_loader(d)
    ret = iterate(dl)
    ret === nothing && return nothing
    obs, state = ret
    return obs, (dl, state)
end
