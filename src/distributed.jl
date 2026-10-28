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
#
# A `nothing` index set is a pre-warm sentinel (see `_distributed_loader`): it primes the
# `CachingPool` — first-touching the worker so `data` is serialized on the caller's task —
# without doing any work. Real `idxsets` elements are always `Int`/`Vector{Int}`, so
# `nothing` is unambiguous.
# `shm` is a `ShmSession` (pack collated batches into shared memory) or `nothing` (serialize as
# before). Packing happens *inside* the closure, on the worker, so only the small skeleton crosses
# `remotecall_fetch`. The `nothing` pre-warm sentinel is untouched (no batch, so nothing to pack).
#
# Distributed runs every remote call with `invokelatest`, so a redefinition landing on a worker
# mid-epoch would leak into it. The batch is therefore run with `invoke_in_world` at the world age
# the worker had when the epoch began; `worlds[i]` is that age for worker `pids[i]` (see `_worker_worlds`).
# This is a flat vector rather than a `Dict` keyed by pid: the `Dict` measurably slowed every call.
function _worker_closure(collate, data, shm, pids)
    return function (worlds, idxs)
        idxs === nothing && return nothing
        world = worlds[findfirst(==(Distributed.myid()), pids)]
        return Base.invoke_in_world(world, _worker_batch, collate, data, idxs, shm)
    end
end

_worker_batch(collate, data, idxs, shm) = shm_pack(_worker_getbatch(collate, data, idxs), shm)

# Current world age of each worker, taken once `iterate` has finished shipping definitions to them.
_worker_worlds(pids) = asyncmap(pid -> remotecall_fetch(Base.get_world_counter, pid), pids)

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
# (Module loading is done by the caller, `_distributed_loader`, where `collate` is in scope.)
function _lease_workers(n::Int, owner)
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
    return pids
end

# Are all of `pids` still alive? (False after `close_dataloader_pool`, so a stale cached
# `CachingPool` gets rebuilt rather than dispatched onto dead workers.)
_pool_alive(pids) = !isempty(pids) && all(in(Distributed.procs()), pids)

# Root package modules referenced by a (possibly nested) type — walk type parameters and
# field types, guarding against cycles. Handles Union/UnionAll wrappers (e.g. an optional
# `Union{Nothing, Dataset}` child field). Excludes Base/Core/Main.
function _referenced_modules!(acc::Set{Module}, seen::Set{Any}, @nospecialize(T))
    (T in seen) && return acc
    push!(seen, T)
    T isa UnionAll && return _referenced_modules!(acc, seen, T.body)
    T isa Union    && (_referenced_modules!(acc, seen, T.a);
                       return _referenced_modules!(acc, seen, T.b))
    T isa DataType || return acc
    let m = Base.moduleroot(parentmodule(T))
        (m === Base || m === Core || m === Main) || push!(acc, m)
    end
    for p in T.parameters
        p isa Type && _referenced_modules!(acc, seen, p)
    end
    # Only types with a definite field layout can be walked; `fieldtypes` throws on abstract
    # types (which are also `DataType`s) and other indefinite types (e.g. an abstractly-typed
    # `jltransform` field reached during the walk). We already recorded such a type's own
    # module above, so skipping its (undefined) fields loses nothing loadable.
    if isconcretetype(T)
        for ft in fieldtypes(T)                 # catches captured closure fields, loose wrappers
            ft isa Type && _referenced_modules!(acc, seen, ft)
        end
    end
    return acc
end

_referenced_modules(@nospecialize(x)) =
    _referenced_modules!(Set{Module}(), Set{Any}(), typeof(x))

# Make MLUtils (needed to deserialize the worker closure and call `getobs`) and every root
# package referenced by the (possibly nested) `data` container and `collate` function
# available on each worker, so it can deserialize them. Runs on the main task, where the
# concrete types are visible — so it sees a `Dataset` nested inside a `MappedData` that the
# outermost-type-only approach would miss.
#
# We only `import` packages that are *directly loadable* in this environment (a direct
# dependency or a stdlib, per `Base.identify_package`). The field-type walk can also surface
# a package's own *transitive* dependencies (e.g. `PythonCall` reached through a `Dataset`'s
# internal `Py` field) — those are not directly importable by name, but are pulled onto the
# worker anyway when their parent package is imported, so skipping them is both necessary
# (the bare `import` would fail) and sufficient (they still resolve for `deserialize`).
function _load_modules_on_workers(data, collate, pids)
    isempty(pids) && return nothing
    Distributed.remotecall_eval(Main, pids, :(import MLUtils))
    _import_packages(union(_referenced_modules(data), _referenced_modules(collate)), pids)
    return nothing
end

# `import` each of `mods` on the workers, skipping MLUtils (already imported) and any package
# that is not *directly loadable* in this environment (a direct dependency or a stdlib, per
# `Base.identify_package`) — a transitive-only package (e.g. `PythonCall` reached through a
# `Dataset`'s internal `Py` field) is pulled in anyway when its parent package is imported.
function _import_packages(mods, pids)
    isempty(pids) && return nothing
    for m in mods
        (m === MLUtils) && continue
        name = nameof(m)
        Base.identify_package(String(name)) === nothing && continue
        try
            Distributed.remotecall_eval(Main, pids, :(import $name))
        catch e
            error("MLUtils: could not auto-load `$name` on worker processes for distributed " *
                  "data loading. Load it manually with `@everywhere using $name` before " *
                  "iterating the DataLoader. Original error: $e")
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Shipping `Main`-defined functions/structs to workers
# ---------------------------------------------------------------------------
#
# `Serialization` sends a *named* `Main` function or a `Main` struct by *reference* (module +
# name); the worker then needs that binding to exist or `deserialize` throws `UndefVarError`.
# (Anonymous closures, by contrast, ship whole — by value.) To make e.g.
# `DataLoader(mapobs(named_fn, custom_struct); num_workers=N)` work without the user manually
# `@everywhere`-defining anything, we walk the `data`/`collate` value graph, recover the
# *source* of every referenced `Main` function/struct — via `CodeTracking`, which reads it
# straight from the defining file, with no Revise and no tracking — and `remotecall_eval` those
# definitions onto the leased workers before any data is shipped (i.e. before the pre-warm).
#
# Covered: named functions (including transitive helpers and keyword methods), the `Main`
# globals those bodies read, `Main` structs used as containers (with their `getobs`/`numobs`/…
# methods), and closures that capture any of the above. Not covered: objects with no
# file-backed source (defined at the REPL or via `eval`/`include_string`) — those raise a
# precise error listing the fixes, instead of a cryptic worker-side `UndefVarError`.

# Source (as an `Expr`) of a method's definition, read from its defining file; `nothing` if
# unavailable (REPL/`eval`-defined). `CodeTracking.definition` resolves keyword methods and
# depot-path remapping for us.
function _method_source(m::Method)
    got = CodeTracking.definition(String, m)
    got === nothing && return nothing
    ex = try Meta.parse(String(first(got)); raise = false) catch; nothing end
    return (ex isa Expr && !Meta.isexpr(ex, :incomplete) && !Meta.isexpr(ex, :error)) ? ex : nothing
end

# Source of a `Main` struct's `struct … end` block. `CodeTracking` targets *methods*, not type
# definitions, so parse the defining file (located via the auto-generated constructor's
# file/line) and take the top-level expression covering that line.
function _struct_source(@nospecialize(T::DataType))
    isempty(methods(T)) && return nothing
    m = first(methods(T))
    path, line = string(m.file), Int(m.line)
    isfile(path) || return nothing
    ast = try Meta.parseall(read(path, String); filename = path) catch; return nothing end
    covering = nothing; curline = 0
    for a in ast.args
        if a isa LineNumberNode
            curline = a.line
        elseif curline <= line
            covering = a                    # last top-level expr at/above the struct's line
        end
    end
    return covering
end

_ismain(@nospecialize(T)) = T isa DataType && T.name.module === Main

# Wrap a struct definition so re-shipping it to a *reused* warm worker (which may already have
# the type — struct redefinition is an error) is a no-op. Functions/globals re-eval harmlessly.
_guard_struct(name::Symbol, ex) =
    :(isdefined(Main, $(QuoteNode(name))) || Core.eval(Main, $(QuoteNode(ex))))

# Walk the value graph of `roots`, collecting the ordered `Main` definitions the workers need
# (`structs`, then `globals`, then `funcs`), the extra packages the shipped bodies call
# (`mods`), and the names of anything whose source could not be recovered (`unresolved`).
function _collect_worker_defs(roots)
    structs = Expr[]; globals = Expr[]; funcs = Expr[]
    mods = Set{Module}(); unresolved = Symbol[]
    done = Set{Symbol}()                    # def-names already emitted
    locs = Set{Tuple{String,Int}}()         # method source locations already shipped
    seen = Base.IdSet{Any}()                # value-graph cycle guard
    visit = Ref{Any}()

    # Ship a method's source once per file location (a keyword method surfaces as several
    # `Method` objects — body + sorter — all pointing at the same definition line).
    pushmethod(m::Method, nm::Symbol) = begin
        loc = (string(m.file), Int(m.line))
        loc in locs && return
        push!(locs, loc)
        ex = _method_source(m)
        ex === nothing ? push!(unresolved, nm) : push!(funcs, ex)
        scanbody(m)
    end

    # Descend into a value's contents (struct fields / tuple / small array elements) to find
    # nested `Main` functions and struct instances. Bits arrays (numbers, etc.) are skipped.
    recurse(@nospecialize(v)) = begin
        if v isa Tuple || v isa NamedTuple
            for x in v; visit[](x); end
        elseif v isa AbstractArray
            (isbitstype(eltype(v)) || eltype(v) <: Union{AbstractString,Symbol}) || for x in v
                visit[](x)
            end
        elseif isstructtype(typeof(v))
            for i in 1:nfields(v)
                isdefined(v, i) || continue
                f = try getfield(v, i) catch; continue end
                visit[](f)
            end
        end
    end

    # Scan a method's lowered code for `GlobalRef`s: `Main` functions/structs → ship them,
    # `Main` plain globals → ship by value, package refs → import that package on the worker.
    scanbody(m::Method) = begin
        ci = try Base.uncompressed_ir(m) catch; return end
        scan = Ref{Any}()
        scan[] = (@nospecialize(x)) -> begin
            if x isa GlobalRef
                if x.mod === Main && isdefined(Main, x.name)
                    v = getglobal(Main, x.name)
                    if v isa Function || v isa Type
                        visit[](v)
                    elseif !(v isa Module)
                        _ismain(typeof(v)) && visit[](v)
                        if !(x.name in done)
                            push!(done, x.name)
                            push!(globals, :(global $(x.name) = $(QuoteNode(v))))
                        end
                    end
                elseif x.mod !== Base && x.mod !== Core && x.mod !== Main
                    push!(mods, Base.moduleroot(x.mod))
                end
            elseif x isa Expr
                foreach(scan[], x.args)
            elseif x isa Core.ReturnNode
                isdefined(x, :val) && scan[](x.val)
            elseif x isa Core.GotoIfNot
                scan[](x.cond)
            end
        end
        foreach(scan[], ci.code)
    end

    # A `Main`-bound named function's Symbol, or `nothing` for anonymous/closure/package fns.
    mainfuncname(@nospecialize(f)) = begin
        T = typeof(f)
        (T isa DataType && T.name.module === Main && supertype(T) === Function) || return nothing
        nm = nameof(f)
        (isdefined(Main, nm) && getglobal(Main, nm) === f) ? nm : nothing
    end

    shipfunc(@nospecialize(f), nm::Symbol) = begin
        nm in done && return
        push!(done, nm)
        for m in methods(f); pushmethod(m, nm); end
    end

    shipstruct(@nospecialize(T::DataType)) = begin
        nm = nameof(T)
        nm in done && return
        push!(done, nm)
        ex = _struct_source(T)
        ex === nothing ? push!(unresolved, nm) : push!(structs, _guard_struct(nm, ex))
        for m in methodswith(T; supertypes = false)
            (m.name === nm || m.module !== Main) && continue   # skip ctor / package methods
            pushmethod(m, m.name)
        end
    end

    visit[] = (@nospecialize(v)) -> begin
        (v isa Symbol || v isa Module) && return
        v in seen && return
        push!(seen, v)
        if v isa Function
            nm = mainfuncname(v)
            if nm === nothing                       # anonymous/closure: ships by value…
                if typeof(v).name.module === Main    # …but scan its body + captured fields
                    foreach(scanbody, methods(v)); recurse(v)
                end
            else
                shipfunc(v, nm)
            end
        elseif v isa Type
            v isa DataType && _ismain(v) && shipstruct(v)
        else
            _ismain(typeof(v)) && shipstruct(typeof(v))
            recurse(v)
        end
    end

    foreach(visit[], roots)
    return (; structs, globals, funcs, mods, unresolved)
end

# Ship every `Main`-defined function/struct/global reachable from `data`/`collate` onto the
# workers (plus any package their bodies call), so a named transform or a script-local struct
# container works under `num_workers` without a manual `@everywhere`. Runs after
# `_load_modules_on_workers` (packages first) and before the pool pre-warm (defs before data).
function _ship_worker_defs(data, collate, pids)
    isempty(pids) && return nothing
    defs = _collect_worker_defs((data, collate))
    _import_packages(defs.mods, pids)
    if !isempty(defs.unresolved)
        names = join(sort(unique(String.(defs.unresolved))), ", ")
        error("MLUtils: cannot ship $(names) to worker processes for `num_workers` loading — " *
              "their source is unavailable (defined at the REPL or via `eval`/`include_string`, " *
              "not a file on disk). Fixes: put them in a file loaded with `include`, move them " *
              "into a package/module your project depends on, or use an anonymous function.")
    end
    # Two rounds so evaluation is interleaved with deserialization: a shipped global may hold a
    # *value* of a shipped `Main` struct, and the whole `:toplevel` expression is deserialized on
    # the worker before any of it runs — so the struct types must already exist from a prior
    # `remotecall_eval`. Round 1 (structs + funcs) is pure source with no embedded instances;
    # round 2 (globals) may embed instances, which now deserialize against the round-1 types.
    code = vcat(defs.structs, defs.funcs)
    isempty(code)         || Distributed.remotecall_eval(Main, pids, Expr(:toplevel, code...))
    isempty(defs.globals) || Distributed.remotecall_eval(Main, pids, Expr(:toplevel, defs.globals...))
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
    # Workers are gone, so no straggler can write; reclaim every shared-memory session dir.
    _shm_close_all()
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
    world = Base.get_world_counter()
    if cache === nothing || !_pool_alive(cache.pids)
        pids = _lease_workers(d.num_workers, d._cache)
        _load_modules_on_workers(d.data, collate, pids)  # import referenced packages…
        _ship_worker_defs(d.data, collate, pids)         # …then ship Main funcs/structs/globals
        cpool = CachingPool(pids)
        # One shared-memory session per cache build, created on main before the pre-warm. The
        # session dir's lifetime is tied to `d._cache` via a finalizer (see `_shm_session_create`);
        # `nothing` (shm off, unsupported platform, or mkdir failure) degrades to serialization.
        shmses = d.shm ? _shm_session_create(d._cache) : nothing
        f = _worker_closure(collate, d.data, shmses, pids)
        # Prime the pool from the consumer's task: first-touch each worker here so `d.data`
        # is serialized on this task (not the GIL-less feeder below) and cached remotely.
        # The feeder then ships only cached refs + index sets — it never serializes `d.data`
        # (never touches Python). `asyncmap` keeps serialization on this task/thread while
        # overlapping the N worker-side rebuilds (deserialize / pickle.loads run concurrently
        # across workers instead of N sequential round-trips). Each of the `length(pids)`
        # concurrent calls does `take!(pool)` on the freshly built `cpool`, draining it so
        # every call lands on a distinct worker → each worker first-touched exactly once.
        asyncmap(_ -> remotecall_fetch(f, cpool, nothing, nothing), 1:length(pids); ntasks = length(pids))
        cache = (pids = pids, cpool = cpool, f = f, shm = shmses, world = world)
        d._cache[] = cache
    elseif cache.world != world
        # Methods were (re)defined on main since the workers were last synced — typically Revise
        # applying an edit — so refresh their copies of the `Main` definitions. `data` stays cached.
        _ship_worker_defs(d.data, collate, cache.pids)
        cache = (; cache..., world)
        d._cache[] = cache
    end
    pids, cpool, f, shmses = cache.pids, cache.cpool, cache.f, cache.shm
    worlds = _worker_worlds(pids)

    nworkers = max(length(pids), 1)
    ch = Channel(nworkers)   # bounded prefetch → overlap + backpressure
    task = Threads.@spawn begin
        try
            # Unpack (mmap + immediate unlink) before `put!`, so the backing files are freed even
            # if `put!` throws on a channel the consumer already closed. Unpacking touches no
            # Python, so keeping it on the feeder task preserves the GIL isolation of the pre-warm.
            asyncmap(idxs -> put!(ch, shm_unpack(remotecall_fetch(f, cpool, worlds, idxs), shmses)), idxsets; ntasks = nworkers)
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
