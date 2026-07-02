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
# `persistent_workers=True`), reused across epochs to amortize startup. Unlike PyTorch —
# where every `DataLoader` spawns its *own* workers — MLUtils keeps a single shared pool
# but hands each *live* loader a **disjoint** block of workers. Two loaders alive at once
# (a train and a validation loader, or a loader iterated inside another's loop) therefore
# never bind to the same processes, so they can't contend for them or double-cache their
# `data` on them. Workers whose owning loader has been garbage-collected return to the warm
# pool and are reused by the next loader instead of being killed.
#
# `_ALL_PIDS` is every managed worker (leased + free). `_LEASES` records, per live loader,
# a *weak* reference to its `_cache` (a per-loader mutable object that lives exactly as long
# as the loader) alongside the pids it owns. The weak ref lets a collected loader's workers
# re-enter the free set: the GC nulls the ref, and the next allocation prunes it. The lock
# makes concurrent first-iterations allocate atomically (no double `addprocs`, and never two
# loaders handed the same pid).
const _POOL_LOCK = ReentrantLock()
const _ALL_PIDS = Int[]
const _LEASES = Tuple{WeakRef,Vector{Int}}[]

# Lease `n` worker processes for `owner`, disjoint from every other live loader's workers;
# spawn more only if fewer than `n` are currently free. Returns the leased pids.
function _lease_workers(n::Int, data, owner)
    pids = lock(_POOL_LOCK) do
        # Prune workers that died, and leases whose loader was garbage-collected (weak ref
        # nulled) or a stale prior lease for this same `owner`; the pids they held re-enter
        # the free set. The complement of what live loaders still hold is the free set.
        filter!(in(Distributed.procs()), _ALL_PIDS)
        filter!(e -> !(e[1].value === nothing || e[1].value === owner), _LEASES)
        leased = Int[]
        for (_, owned) in _LEASES
            append!(leased, owned)
        end
        free = setdiff(_ALL_PIDS, leased)
        if length(free) < n
            # Match the parent's environment so workers instantiate the same project (and,
            # for HF datasets, the same CondaPkg Python). `active_project()` is usually a path.
            proj = Base.active_project()
            exeflags = proj === nothing ? `` : `--project=$proj`
            newpids = addprocs(n - length(free); exeflags)
            append!(_ALL_PIDS, newpids)
            append!(free, newpids)
        end
        mine = free[1:n]
        push!(_LEASES, (WeakRef(owner), mine))
        return mine
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
        isempty(_ALL_PIDS) || rmprocs(_ALL_PIDS...)
        empty!(_ALL_PIDS)
        empty!(_LEASES)
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

    # Cache this loader's own `CachingPool` (over its leased, disjoint workers) + the
    # closure, so that `data` stays cached on the workers across epochs (only the shuffled
    # index sets change). Rebuild if those workers were torn down (e.g. by
    # `close_dataloader_pool`) — checked against the cached pids, not the shared global set.
    # `d._cache` is the loader-lifetime key under which the lease is held, so the workers
    # are released back to the pool automatically once this loader is garbage-collected.
    cache = d._cache[]
    if cache === nothing || !_pool_alive(cache.pids)
        pids = _lease_workers(d.num_workers, d.data, d._cache)
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
