using Distributed
using Serialization
import SparseArrays
using SparseArrays: sprand, SparseMatrixCSC
using MLUtils: _referenced_modules

# A wrapper storing its child in an optional (`Union{Nothing, ...}`) field, to exercise the
# Union branch and the field-type walk of `_referenced_modules`. Its own module is `Main`
# (excluded), so any package in the result must have been reached *through* the child field.
struct _OptionalChild
    child::Union{Nothing,SparseMatrixCSC{Float64,Int}}
end

# `Main`-defined data container + helpers for the worker-defs shipping tests. Defined at the
# top level of this file (which has a real path), so `CodeTracking` can recover their source
# and MLUtils ships it to workers automatically — no `@everywhere` needed. `getobs`/`numobs`
# are extended here (in `Main`) too, so they must ride along with the struct.
struct ShipObs
    base::Int
end
MLUtils.numobs(::ShipObs) = 10
MLUtils.getobs(c::ShipObs, i::Int) = i + c.base
MLUtils.getobs(c::ShipObs, idx::AbstractVector) = [i + c.base for i in idx]

const _SHIP_SCALE = 100
_ship_scale(x) = x * _SHIP_SCALE            # reads the `Main` global `_SHIP_SCALE`
_ship_named(x) = _ship_scale(x) + 1         # calls a transitive `Main` helper
_ship_kw(x; add = 7) = x + add              # keyword method (surfaces as several `Method`s)

@testset "_referenced_modules (Change 1 type walk)" begin
    # Base/Core types carry no package to load.
    @test isempty(_referenced_modules([1, 2, 3]))
    @test isempty(_referenced_modules(rand(4, 4)))
    @test isempty(_referenced_modules((rand(3), collect(1:3))))

    # A nested container: SparseArrays lives inside `MappedData`'s `.data` field, so the walk
    # must descend past the outermost `MLUtils` type to find it (the exact `mapobs(f, ds)`
    # shape that the old outermost-only loader missed).
    mods = _referenced_modules(mapobs(x -> x, sprand(5, 5, 0.5)))
    @test SparseArrays in mods
    @test MLUtils in mods

    # Optional (`Union{Nothing, T}`) field: reached only via the Union + field-type branches.
    @test _referenced_modules(_OptionalChild(nothing)) == Set([SparseArrays])

    # Walking `collate` too: a collate closure capturing a package type pulls that package in.
    sp = sprand(3, 3, 0.5)
    @test SparseArrays in _referenced_modules(_ -> sp)
end

@testset "argument validation (no workers needed)" begin
    Z = rand(4, 10)
    # parallel and num_workers are mutually exclusive
    @test_throws ArgumentError DataLoader(Z; parallel=true, num_workers=2)
    @test_throws ArgumentError DataLoader(Z; parallel=2, num_workers=2)
    # buffer is not supported with num_workers
    @test_throws ArgumentError DataLoader(Z; num_workers=2, buffer=true)
    # num_workers=0 stays on the serial path, byte-for-byte the old behavior
    @test typeof(DataLoader(Z)).parameters[3] === :serial
    @test typeof(DataLoader(Z; num_workers=0)).parameters[3] === :serial
    @test typeof(DataLoader(Z; num_workers=2)).parameters[3] === :distributed
    # num_workers shows up in printing only when set
    @test !occursin("num_workers", repr(DataLoader(Z)))
    @test occursin("num_workers=2", repr(DataLoader(Z; num_workers=2)))
end

@testset "distributed loading over arrays" begin
    # deterministic data so we can check content (order-agnostic, as ordering is
    # not guaranteed under num_workers, exactly as for parallel).
    D = reshape(collect(1.0:80.0), 4, 20)

    @testset "collate=$(coll)" for coll in (Val(nothing), true, false)
        dl = DataLoader(D; batchsize=5, num_workers=2, collate=coll)
        @test length(dl) == 4
        batches = collect(dl)
        @test length(batches) == 4

        if coll === false
            # each batch is a vector of per-observation columns
            cols = [c for b in batches for c in b]
            @test sort([sum(c) for c in cols]) == sort([sum(c) for c in eachcol(D)])
        else
            @test all(b -> size(b) == (4, 5), batches)
            got = sort([sum(c) for b in batches for c in eachcol(b)])
            @test got == sort([sum(c) for c in eachcol(D)])
        end
    end

    # batchsize <= 0: iterate single observations (ObsView path, no collate)
    dl = DataLoader(D; batchsize=-1, num_workers=2)
    obs = collect(dl)
    @test length(obs) == 20
    @test sort([sum(c) for c in obs]) == sort([sum(c) for c in eachcol(D)])

    # tuples + shuffle: every observation shows up exactly once
    dl = DataLoader((D, collect(1:20)); batchsize=4, num_workers=2, shuffle=true, collate=true)
    batches = collect(dl)
    @test length(batches) == 5
    labels = sort(reduce(vcat, [b[2] for b in batches]))
    @test labels == collect(1:20)

    # a second epoch reuses the loader's own warm workers + cached data
    nprocs_before = nprocs()
    @test length(collect(dl)) == 5
    @test nprocs() == nprocs_before   # no new workers spawned

    # a *distinct* loader, while `dl` is still alive, gets its OWN disjoint workers so
    # concurrent/nested loaders never contend for the same processes
    dl2 = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dl2)) == 4
    @test isempty(intersect(dl._cache[].pids, dl2._cache[].pids))

    # early termination (`first`) yields a valid batch and does not hang
    @test size(first(DataLoader(D; batchsize=5, num_workers=2))) == (4, 5)

    # errors thrown inside a worker propagate to the consumer
    bad = mapobs(_ -> error("boom in worker"), collect(1:8))
    dlbad = DataLoader(bad; batchsize=2, num_workers=2)
    @test_throws Exception collect(dlbad)
end

# Reproduces the design's §8(b): a container with a custom, recipe-based
# `Serialization` dispatched over MLUtils' `CachingPool` is reconstructed exactly
# ONCE per worker (not once per batch), so `data` crosses the process boundary once.
@testset "reconstruct-once per worker (CachingPool)" begin
    MLUtils.close_dataloader_pool()
    # Force the managed pool to exist so `@everywhere` can define the toy type on it. The
    # throwaway loader is created inside a function so it becomes unreachable on return; a
    # GC below then frees its workers back to the pool and the real loader re-leases those
    # same (now type-aware) workers rather than spawning fresh, type-less ones.
    _warmup(n) = (collect(DataLoader(rand(2, 6); batchsize=3, num_workers=n)); nothing)
    _warmup(2)
    @test length(workers()) >= 2

    @everywhere begin
        using Serialization
        import MLUtils
        const _RECON = Ref(0)
        struct ReconCounter
            n::Int
        end
        MLUtils.numobs(r::ReconCounter) = r.n
        MLUtils.getobs(r::ReconCounter, i) = i     # stand-in for mmap+decode; returns plain data
        Serialization.serialize(s::AbstractSerializer, r::ReconCounter) =
            (Serialization.serialize_type(s, ReconCounter); Serialization.serialize(s, r.n))
        function Serialization.deserialize(s::AbstractSerializer, ::Type{ReconCounter})
            n = Serialization.deserialize(s)
            _RECON[] += 1                          # count reconstructions on the worker
            return ReconCounter(n)
        end
        _reconcount() = _RECON[]
        _reset_recon() = (_RECON[] = 0)
    end

    foreach(w -> remotecall_fetch(_reset_recon, w), workers())

    # Release the throwaway loader's lease so the real loader reuses those workers
    # (which now have `ReconCounter` defined) instead of spawning fresh, type-less ones.
    GC.gc(); GC.gc()

    data = ReconCounter(100)
    dl = DataLoader(data; batchsize=10, num_workers=2, collate=false)  # 10 batches
    batches = collect(dl)
    # results are correct and order-agnostic
    @test sort(reduce(vcat, batches)) == collect(1:100)

    counts = [remotecall_fetch(_reconcount, w) for w in workers()]
    # reconstructed once per worker that did work, never once-per-batch
    @test sum(counts) == count(>(0), counts)
    @test sum(counts) <= length(workers())
    @test sum(counts) >= 1
end

@testset "concurrent loaders get disjoint workers (no contention)" begin
    MLUtils.close_dataloader_pool()
    D = reshape(collect(1.0:60.0), 4, 15)

    dlA = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dlA)) == 3
    @test nprocs() == 3                      # main + 2
    pidsA = copy(dlA._cache[].pids)
    @test length(pidsA) == 2

    # a second loader created while `dlA` is still alive must get its OWN, disjoint
    # workers — the fix for the bug where distinct loaders shared (and fought over) pids
    dlB = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dlB)) == 3
    pidsB = copy(dlB._cache[].pids)
    @test length(pidsB) == 2
    @test isempty(intersect(pidsA, pidsB))   # disjoint: no shared workers
    @test nprocs() == 5                      # main + 2 + 2 (grew rather than sharing)

    # each loader re-iterates on its own warm workers across epochs, no growth
    @test length(collect(dlA)) == 3
    @test dlA._cache[].pids == pidsA
    @test length(collect(dlB)) == 3
    @test dlB._cache[].pids == pidsB
    @test nprocs() == 5
end

@testset "a dropped loader's workers return to the pool" begin
    MLUtils.close_dataloader_pool()
    D = reshape(collect(1.0:40.0), 4, 10)

    # run a loader entirely inside a function so it is unreachable after the call returns
    run_once() = (@test length(collect(DataLoader(D; batchsize=5, num_workers=2))) == 2; nothing)
    run_once()
    @test nprocs() == 3                      # main + 2
    GC.gc(); GC.gc()                         # collect the loader; its lease is released

    # a new same-size loader reuses the freed workers instead of spawning more
    dl2 = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dl2)) == 2
    @test nprocs() == 3                      # reused, no new addprocs
end

@testset "pool teardown rebuilds a stale loader" begin
    D = reshape(collect(1.0:40.0), 4, 10)
    dl = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dl)) == 2
    MLUtils.close_dataloader_pool()
    @test nprocs() == 1
    # re-iterating the same loader after teardown must rebuild the pool, not crash
    batches = collect(dl)
    @test length(batches) == 2
    @test sort([sum(c) for b in batches for c in eachcol(b)]) == sort([sum(c) for c in eachcol(D)])
end

# Change 2: pre-warming the CachingPool on the consumer's task caches the closure `f`
# (and the `data` it captures) exactly once per worker. This asserts a stdlib internal
# (`CachingPool.map_obj2ref`) — a loose smoke check, relaxable if its layout changes.
@testset "pre-warm caches once per worker (CachingPool)" begin
    MLUtils.close_dataloader_pool()
    D = reshape(collect(1.0:40.0), 4, 10)
    dl = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dl)) == 2
    cpool = dl._cache[].cpool
    # one cached `(worker, f)` entry per leased worker — data crossed the wire once each
    @test length(cpool.map_obj2ref) == length(dl._cache[].pids) == 2
    # a second epoch reuses the same warm pool: no re-lease, no extra cache entries
    nprocs_before = nprocs()
    @test length(collect(dl)) == 2
    @test nprocs() == nprocs_before
    @test length(cpool.map_obj2ref) == 2
    MLUtils.close_dataloader_pool()
end

# Worker-defs shipping: named `Main` functions/structs (and the globals/helpers/captures they
# pull in) are auto-shipped to workers via their recovered source, so they work under
# `num_workers` without a manual `@everywhere`.
@testset "ship Main defs to workers" begin
    @testset "collect" begin
        defs = MLUtils._collect_worker_defs((mapobs(_ship_named, ShipObs(1)), _ship_kw))
        @test isempty(defs.unresolved)          # everything is file-backed
        @test !isempty(defs.structs)            # ShipObs
        @test !isempty(defs.globals)            # _SHIP_SCALE
        @test !isempty(defs.funcs)              # _ship_named, _ship_scale, getobs/numobs, _ship_kw
        # a function with no backing source file → reported as unresolved
        include_string(Main, "_no_source_fn(x) = x", "not_a_real_file.jl")
        d2 = MLUtils._collect_worker_defs((mapobs(_no_source_fn, [1,2]), Val(nothing)))
        @test :_no_source_fn in d2.unresolved
    end

    @testset "named function (+ transitive helper + global)" begin
        MLUtils.close_dataloader_pool()
        dl = DataLoader(mapobs(_ship_named, collect(1:10)); batchsize=5, num_workers=2, collate=true)
        @test sort(reduce(vcat, collect(dl))) == sort([x * 100 + 1 for x in 1:10])
    end

    @testset "keyword method" begin
        MLUtils.close_dataloader_pool()
        dl = DataLoader(mapobs(_ship_kw, collect(1:8)); batchsize=4, num_workers=2, collate=true)
        @test sort(reduce(vcat, collect(dl))) == sort([x + 7 for x in 1:8])
    end

    @testset "Main-struct container (+ its getobs/numobs)" begin
        MLUtils.close_dataloader_pool()
        dl = DataLoader(ShipObs(20); batchsize=5, num_workers=2, collate=false)
        @test sort(reduce(vcat, collect(dl))) == collect(21:30)
    end

    @testset "named function over a Main-struct container" begin
        MLUtils.close_dataloader_pool()
        dl = DataLoader(mapobs(_ship_named, ShipObs(0)); batchsize=5, num_workers=2, collate=true)
        @test sort(reduce(vcat, collect(dl))) == sort([i * 100 + 1 for i in 1:10])
    end

    @testset "closure capturing a named function and a Main-struct instance" begin
        MLUtils.close_dataloader_pool()
        inst = ShipObs(3)                                    # local → captured by value
        cl = x -> _ship_scale(x) + MLUtils.getobs(inst, 1)   # captures inst; calls named _ship_scale
        dl = DataLoader(mapobs(cl, collect(1:6)); batchsize=3, num_workers=2, collate=true)
        @test sort(reduce(vcat, collect(dl))) == sort([x * 100 + (1 + 3) for x in 1:6])
    end

    @testset "unshippable (REPL/eval-defined) errors clearly" begin
        MLUtils.close_dataloader_pool()
        include_string(Main, "_repl_fn(x) = x + 1", "no_such_file.jl")   # no backing source file
        dl = DataLoader(mapobs(_repl_fn, collect(1:4)); batchsize=2, num_workers=2, collate=true)
        @test_throws "source is unavailable" collect(dl)
    end

    MLUtils.close_dataloader_pool()
end

MLUtils.close_dataloader_pool()
@test nprocs() == 1
