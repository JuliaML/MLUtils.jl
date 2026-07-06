# Unit tests for the shared-memory batch transport (`src/shmem.jl`). These run in-process with a
# hand-built session in a temp dir — no worker processes needed — plus the validation/printing
# tests for the `shm` kwarg. The distributed integration tests live in `test/distributed.jl`.

using MLUtils
using MLUtils: ShmSession, ShmHandle, ShmVector, ShmDict, shm_pack, shm_unpack,
               _shm_supported, _shm_packable, _shm_session_create, _shm_close_all,
               _shm_dir_pid, _pid_alive, _shm_root, _SHM_MIN_BYTES, _SHM_SESSIONS

# A small session whose threshold is low enough that the modest test arrays pack.
withsession(f; minbytes = 64) = mktempdir() do dir
    f(ShmSession(dir, minbytes))
end

@testset "leaf predicate" begin
    withsession() do ses
        @test _shm_packable(rand(4, 100), ses.minbytes)             # large dense array
        @test !_shm_packable(rand(2), ses.minbytes)                 # below threshold
        @test !_shm_packable(1:10_000, ses.minbytes)                # range: O(1)-serializable
        @test _shm_packable(collect(1:10_000)', ses.minbytes)       # adjoint is packable
        @test _shm_packable(view(rand(10, 50), :, :), ses.minbytes) # large view is packable
        @test !_shm_packable([rand(3) for _ in 1:9], ses.minbytes)  # non-bits eltype
    end
end

@testset "pack/unpack round-trip" begin
    withsession() do ses
        # dense Array
        A = rand(4, 100)
        h = shm_pack(A, ses)
        @test h isa ShmHandle
        B = shm_unpack(h, ses)
        @test B == A && B isa Array{Float64,2}

        # tuple (mixed pack + passthrough)
        t = (rand(Float32, 8, 8, 16), 1.5)
        st = shm_pack(t, ses)
        @test st[1] isa ShmHandle && st[2] === 1.5
        @test shm_unpack(st, ses)[1] == t[1]

        # NamedTuple with a small label riding the skeleton
        nt = (image = rand(Float32, 16, 16, 8), label = Int8[1, 2, 3])
        snt = shm_pack(nt, ses)
        @test snt.image isa ShmHandle && snt.label == nt.label     # label below threshold
        rnt = shm_unpack(snt, ses)
        @test rnt.image == nt.image && rnt.label == nt.label

        # single obs (a lone array)
        o = rand(50, 50)
        @test shm_unpack(shm_pack(o, ses), ses) == o

        # nested container
        nested = (a = (rand(40, 40), rand(40, 40)), b = 7)
        rn = shm_unpack(shm_pack(nested, ses), ses)
        @test rn.a[1] == nested.a[1] && rn.a[2] == nested.a[2] && rn.b == 7
    end
end

@testset "Dict type fidelity" begin
    withsession() do ses
        d1 = Dict("x" => rand(50, 50), "y" => rand(50, 50))
        r1 = shm_unpack(shm_pack(d1, ses), ses)
        @test r1 isa Dict{String,Matrix{Float64}}     # value type preserved (not narrowed/widened)
        @test r1["x"] == d1["x"] && r1["y"] == d1["y"]

        d2 = Dict{String,Any}("x" => rand(50, 50), "n" => 3)
        r2 = shm_unpack(shm_pack(d2, ses), ses)
        @test r2 isa Dict{String,Any}                 # never silently narrows a Dict{String,Any}
        @test r2["x"] == d2["x"] && r2["n"] == 3
    end
end

@testset "ragged Vector round-trip" begin
    withsession() do ses
        # Vector{Matrix}: collate=false vector-of-observations
        vv = [rand(30, 30) for _ in 1:5]
        sv = shm_pack(vv, ses)
        @test sv isa ShmVector
        rvv = shm_unpack(sv, ses)
        @test rvv isa Vector{Matrix{Float64}} && all(rvv .== vv)

        # Vector{Tuple}
        vt = [(rand(20, 20), i) for i in 1:4]
        rvt = shm_unpack(shm_pack(vt, ses), ses)
        @test rvt isa Vector && all(rvt[i][1] == vt[i][1] && rvt[i][2] == vt[i][2] for i in 1:4)
    end
end

@testset "policy: passthrough & normalization" begin
    withsession() do ses
        # below threshold ⇒ identical object (zero-overhead passthrough)
        small = rand(3)
        @test shm_pack(small, ses) === small

        # ranges / non-bits are untouched
        @test shm_pack(1:10_000, ses) === (1:10_000)

        # view / BitArray → dense Array on the receiver (documented normalization)
        V = view(rand(10, 50), :, :)
        @test shm_unpack(shm_pack(V, ses), ses) isa Array

        bits = trues(200)
        rb = shm_unpack(shm_pack(bits, ses), ses)
        @test rb isa Array{Bool} && rb == bits
    end
end

@testset "unpacked arrays are writable (r+ regression)" begin
    withsession() do ses
        A = rand(4, 50)
        B = shm_unpack(shm_pack(A, ses), ses)
        B[2, 3] = -99.0                 # would segfault on a read-only (PROT_READ) mapping
        @test B[2, 3] == -99.0
    end
end

@testset "leak hygiene" begin
    withsession() do ses
        h = shm_pack(rand(60, 60), ses)
        @test length(readdir(ses.dir)) == 1        # one leaf file written
        B = shm_unpack(h, ses)
        @test isempty(readdir(ses.dir))            # unpack unlinks immediately
        @test B[1, 1] == B[1, 1]                   # array still valid after unlink
    end
end

@testset "threshold boundary" begin
    mktempdir() do dir
        ses = ShmSession(dir, 8 * sizeof(Float64))   # threshold = 64 bytes = 8 Float64
        just_below = zeros(7)                         # 56 bytes < threshold ⇒ passthrough
        @test shm_pack(just_below, ses) === just_below
        @test shm_pack(zeros(8), ses) isa ShmHandle   # 64 bytes == threshold ⇒ packed
    end
end

@testset "write failure ⇒ inline fallback + warn" begin
    mktempdir() do dir
        chmod(dir, 0o500)                          # read/exec only: leaf create fails
        ses = ShmSession(dir, 64)
        local got
        @test_logs (:warn,) match_mode = :any (got = shm_pack(rand(4, 50), ses))
        @test got isa Array && !(got isa ShmHandle)   # raw array returned inline, no hard error
        chmod(dir, 0o700)
    end
end

@testset "no-op when there is no session" begin
    A = rand(4, 100)
    @test shm_pack(A, nothing) === A
    @test shm_unpack(A, nothing) === A
end

@testset "stale sweep" begin
    if _shm_supported()
        root = _shm_root()
        # a dir owned by a definitely-dead pid must be swept; our own must survive
        deadpid = 999_999_999
        dead = joinpath(root, "jl_mlutils_shm_$(deadpid)_1_deadbeef")
        mkpath(dead)
        owner = Ref{Any}(nothing)
        ses = _shm_session_create(owner)                # runs the sweep, creates our own dir
        @test ses !== nothing
        @test !isdir(dead)                              # dead-pid dir swept
        @test isdir(ses.dir)                            # our own dir survives
        @test _shm_dir_pid("jl_mlutils_shm_$(deadpid)_1_deadbeef") == deadpid
        @test _pid_alive(Int(getpid()))                 # this process is alive
        _shm_close_all()
        @test isempty(filter(n -> startswith(n, "jl_mlutils_shm_"), readdir(root)))
    end
end

@testset "validation & printing (no workers)" begin
    Z = rand(4, 10)
    # shm=true requires num_workers>0
    @test_throws ArgumentError DataLoader(Z; shm=true)
    # auto: off on the serial path, on under num_workers where supported
    @test DataLoader(Z).shm == false
    @test DataLoader(Z; num_workers=2).shm == _shm_supported()
    @test DataLoader(Z; num_workers=2, shm=false).shm == false
    # printing: default hidden, deviation shown
    @test !occursin("shm", repr(DataLoader(Z; num_workers=2)))
    @test occursin("shm=false", repr(DataLoader(Z; num_workers=2, shm=false)))
    # mapobs preserves shm
    dm = mapobs(x -> x, DataLoader(Z; batchsize=2, num_workers=2, shm=false, collate=true))
    @test dm.shm == false
end
