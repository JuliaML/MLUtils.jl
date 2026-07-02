using Distributed
using Serialization

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

    # a second epoch reuses the warm pool + cached data
    nprocs_before = nprocs()
    @test length(collect(dl)) == 5
    @test nprocs() == nprocs_before   # no new workers spawned

    # a *distinct* loader also reuses the persistent pool (no extra addprocs)
    dl2 = DataLoader(D; batchsize=5, num_workers=2)
    @test length(collect(dl2)) == 4
    @test nprocs() == nprocs_before

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
    # Force the managed pool to exist so `@everywhere` can define the toy type on it.
    collect(DataLoader(rand(2, 6); batchsize=3, num_workers=2))
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

MLUtils.close_dataloader_pool()
@test nprocs() == 1
