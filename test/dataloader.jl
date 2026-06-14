
@testset "DataLoader" begin
    X2 = reshape([1:10;], (2, 5))
    Y2 = [1:5;]

    d = DataLoader(X2, batchsize=2)
    @test @inferred(first(d)) isa Array
    batches = collect(d)
    @test_broken  eltype(d) == typeof(X2)
    @test eltype(batches) == typeof(X2)
    @test length(batches) == length(d) == 3
    @test size(batches) == size(d) == (3,)
    @test batches[1] == X2[:,1:2]
    @test batches[2] == X2[:,3:4]
    @test batches[3] == X2[:,5:5]

    d = DataLoader(X2, batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    @test_broken eltype(d) == typeof(X2)
    @test length(batches) == 2
    @test batches[1] == X2[:,1:2]
    @test batches[2] == X2[:,3:4]

    d = DataLoader((X2,), batchsize=2, partial=false)
    @inferred first(d)
    batches = collect(d)
    @test_broken eltype(d) == Tuple{typeof(X2)}
    @test eltype(batches) == Tuple{typeof(X2)}
    @test length(batches) == 2
    @test batches[1] == (X2[:,1:2],)
    @test batches[2] == (X2[:,3:4],)

    d = DataLoader((X2, Y2), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test_broken eltype(d) == Tuple{typeof(X2), typeof(Y2)}
    @test eltype(batches) == Tuple{typeof(X2), typeof(Y2)}
    @test length(batches) == 3
    @test length(batches[1]) == 2
    @test length(batches[2]) == 2
    @test length(batches[3]) == 2
    @test batches[1][1] == X2[:,1:2]
    @test batches[1][2] == Y2[1:2]
    @test batches[2][1] == X2[:,3:4]
    @test batches[2][2] == Y2[3:4]
    @test batches[3][1] == X2[:,5:5]
    @test batches[3][2] == Y2[5:5]

    # test with NamedTuple
    d = DataLoader((x=X2, y=Y2), batchsize=2)
    @inferred first(d)
    batches = collect(d)
    @test_broken eltype(d) == NamedTuple{(:x, :y), Tuple{typeof(X2), typeof(Y2)}}
    @test eltype(batches) == NamedTuple{(:x, :y), Tuple{typeof(X2), typeof(Y2)}}
    @test length(batches) == 3
    @test length(batches[1]) == 2
    @test length(batches[2]) == 2
    @test length(batches[3]) == 2
    @test batches[1][1] == batches[1].x == X2[:,1:2]
    @test batches[1][2] == batches[1].y == Y2[1:2]
    @test batches[2][1] == batches[2].x == X2[:,3:4]
    @test batches[2][2] == batches[2].y == Y2[3:4]
    @test batches[3][1] == batches[3].x == X2[:,5:5]
    @test batches[3][2] == batches[3].y == Y2[5:5]

    @testset "iteration default batchsize (+1)" begin
        # test iteration
        X3 = zeros(2, 10)
        d  = DataLoader(X3)
        for x in d
            @test size(x) == (2,1)
        end

        # test iteration
        X3 = ones(2, 10)
        Y3 = fill(5, 10)
        d  = DataLoader((X3, Y3))
        for (x, y) in d
            @test size(x) == (2,1)
            @test y == [5]
        end
    end

    @testset "partial=false" begin
        x = [1:12;]
        d = DataLoader(x, batchsize=5, partial=false) |> collect
        @test length(d) == 2
        @test d[1] == 1:5
        @test d[2] == 6:10
    end

    @testset "shuffle & rng" begin
        X4 = rand(2, 1000)
        d1 = DataLoader(X4, batchsize=2; shuffle=true)
        d2 = DataLoader(X4, batchsize=2; shuffle=true)
        @test first(d1) != first(d2)
        Random.seed!(17)
        d1 = DataLoader(X4, batchsize=2; shuffle=true)
        x1 = first(d1)
        Random.seed!(17)
        d2 = DataLoader(X4, batchsize=2; shuffle=true)
        @test x1 == first(d2)
        d1 = DataLoader(X4, batchsize=2; shuffle=true, rng=MersenneTwister(1))
        d2 = DataLoader(X4, batchsize=2; shuffle=true, rng=MersenneTwister(1))
        @test first(d1) == first(d2)
    end


    # numobs/getobs compatibility
    d = DataLoader(CustomType(), batchsize=2)
    @test first(d) == [1, 2]
    @test length(collect(d)) == 8

    @testset "Dict" begin
        data = Dict("x" => rand(2,4), "y" => rand(4))
        dloader = DataLoader(data, batchsize=2)
        @test_broken eltype(dloader) == Dict{String, Array{Float64}}
        c = collect(dloader)
        @test eltype(c) == Dict{String, Array{Float64}}
        @test c[1] == Dict("x" => data["x"][:,1:2], "y" => data["y"][1:2])
        @test c[2] == Dict("x" => data["x"][:,3:4], "y" => data["y"][3:4])

        data = Dict("x" => rand(2,4), "y" => rand(2,4))
        dloader = DataLoader(data, batchsize=2)
        @test_broken eltype(dloader) == Dict{String, Matrix{Float64}}
        @test eltype(collect(dloader)) == Dict{String, Matrix{Float64}}
    end


    @testset "range" begin
        data = 1:10

        dloader = DataLoader(data, batchsize=2)
        c = collect(dloader)
        @test eltype(c) == Vector{Int64}
        @test c[1] == 1:2

        dloader = DataLoader(data, batchsize=2, shuffle=true)
        c = collect(dloader)
        @test eltype(c) == Vector{Int}
    end

    # https://github.com/FluxML/Flux.jl/issues/1935
    @testset "no views of arrays" begin
        x = CustomArrayNoView(6)
        @test_throws ErrorException view(x, 1:2)

        d = DataLoader(x)
        @test length(collect(d)) == 6 # succesfull iteration

        d = DataLoader(x, batchsize=2, shuffle=false)
        @test length(collect(d)) == 3 # succesfull iteration

        d = DataLoader(x, batchsize=2, shuffle=true)
        @test length(collect(d)) == 3 # succesfull iteration
    end

    @testset "collating" begin
        X_ = rand(10, 20)

        d = DataLoader(X_, collate=false, batchsize = 2)
        @inferred first(d)
        for (i, x) in enumerate(d)
            @test x == [getobs(X_, 2i-1), getobs(X_, 2i)]
        end

        d = DataLoader(X_, collate=nothing, batchsize = 2)
        @inferred first(d)
        for (i, x) in enumerate(d)
            @test x == hcat(getobs(X_, 2i-1), getobs(X_, 2i))
        end

        d = DataLoader(X_, collate=true, batchsize = 2)
        @inferred first(d)
        for (i, x) in enumerate(d)
            @test x == hcat(getobs(X_, 2i-1), getobs(X_, 2i))
        end

        d = DataLoader((X_, X_), collate=false, batchsize = 2)
        @inferred first(d)
        for (i, x) in enumerate(d)
            @test x isa Vector
            all((isa).(x, Tuple))
        end

        d = DataLoader((X_, X_), collate=true, batchsize = 2)
        @inferred first(d)
        for (i, x) in enumerate(d)
            @test all(==(hcat(getobs(X_, 2i-1), getobs(X_, 2i))), x)
        end

        @testset "nothing vs. true" begin
            d = CustomRangeIndex(10)
            @test first(DataLoader(d, batchsize = 2, collate=nothing)) isa Vector
            @test first(DataLoader(d, batchsize = 2, collate=true)) isa Vector
        end
    end

    @testset "collate function" begin
        function collate_fn(batch)
            # collate observations into a custom batch
            return hcat([x[1] for x in batch]...), join([x[2] for x in batch])
        end

        loader = DataLoader((rand(10, 4), ["a", "b", "c", "d"]), batchsize=2, collate=collate_fn)
        for (x, y) in loader
            @test size(x) == (10, 2)
            @test y isa String
        end

        @test first(loader)[2] == "ab"
    end
    
    @testset "mapobs" begin
        X = ones(3, 6)

        function f_mapobs(x)
            return sum(x[1])
        end

        d = DataLoader(X, batchsize=2, collate=false);

        d = mapobs(f_mapobs, d);

        for x in d
            @test x == 3
        end

        d2 = DataLoader(X, batchsize=2, collate=true);

        function f2_mapobs(x)
            return sum(x)
        end

        d2 = mapobs(f2_mapobs, d2);

        for x in d2
           @test x == 6
        end
    end

    if VERSION > v"1.10"
        @testset "printing" begin
            X2 = reshape(Float32[1:10;], (2, 5))
            Y2 = [1:5;]

            d = DataLoader((X2, Y2), batchsize=3)
            
            @test contains(repr(d), "DataLoader(data::Tuple{Matrix")
            @test contains(repr(d), "batchsize=3")

            @test contains(repr(MIME"text/plain"(), d), "2-element DataLoader")
            @test contains(repr(MIME"text/plain"(), d), "2×3 Matrix{Float32}, 3-element Vector")
            
            d2 = DataLoader((x = X2, y = Y2), batchsize=2, partial=false)

            @test contains(repr(d2), "DataLoader(data::@NamedTuple")
            @test contains(repr(d2), "partial=false")

            @test contains(repr(MIME"text/plain"(), d2), "2-element DataLoader(data::@NamedTuple")
            @test contains(repr(MIME"text/plain"(), d2), "x = 2×2 Matrix{Float32}, y = 2-element Vector")
        end
    end

    @testset "buffer issue 205" begin

        function shift_pair(X)
            inputs = map(X) do x
                T = size(x, 4)
                return selectdim(x, 4, 1:(T-1))
            end 
            targets = map(X) do x
                T = size(x, 4)
                return selectdim(x, 4, 2:T)
            end 
            return (stack(inputs), stack(targets))
        end

        trajectory = randn(Float32, 32, 32, 4, 3, 5); 

        loader = DataLoader(
            trajectory;
            batchsize=2,
            partial=false,
            buffer=true,
            collate = shift_pair,
            shuffle = false,
        )

        @test first(loader)[1] == trajectory[:, :, :, 1:2, 1:2]
        @test first(loader)[2] == trajectory[:, :, :, 2:3, 1:2]
    end

    @testset "buffer + collate + parallel issue 216" begin
        # `buffer=true` together with `collate=true` and `parallel=true` used to
        # error because the collated batch has a different type than the per-obs
        # buffer it was built from.
        X_ = rand(Float32, 4, 15)
        serial = collect(DataLoader(X_; batchsize=2, buffer=true, collate=true, parallel=false))
        par    = DataLoader(X_; batchsize=2, buffer=true, collate=true, parallel=true)
        @test_nowarn for _ in par end
        batches = collect(par)
        @test all(b -> b isa Matrix{Float32} && size(b, 1) == 4, batches)
        # parallel order is not guaranteed, so compare as a set of columns
        cols(bs) = sort(collect(eachcol(reduce(hcat, bs))), by=first)
        @test cols(batches) == cols(serial)

        # custom collate function over a non-array container
        loader = DataLoader(["a", "b", "c", "d"]; batchsize=2, buffer=true,
                            collate=join, parallel=true)
        @test sort(collect(loader)) == ["ab", "cd"]
    end
end

@testset "eachobs" begin
    for (i,x) in enumerate(eachobs(X))
        @test x == X[:,i]
    end

    for (i,x) in enumerate(eachobs(X, buffer=true))
        @test x == X[:,i]
    end

    b = zeros(size(X, 1))
    for (i,x) in enumerate(eachobs(X, buffer=b))
        @test x == X[:,i]
    end
    @test b == X[:,end]

    @testset "batched" begin
        for (i, x) in enumerate(eachobs(X, batchsize=2, partial=true))
            if i != 8
                @test size(x) == (4,2)
                @test x == X[:,2i-1:2i]
            else
                @test size(x) == (4,1)
                @test x == X[:,2i-1:2i-1]
            end
        end

        for (i, x) in enumerate(eachobs(X, batchsize=2, buffer=true, partial=false))
            @test size(x) == (4,2)
            @test x == X[:,2i-1:2i]
        end

        b = zeros(4, 2)
        for (i, x) in enumerate(eachobs(X, batchsize=2, buffer=b, partial=false))
            @test size(x) == (4,2)
            @test x == X[:,2i-1:2i]
        end
        @test b == X[:,end-2:end-1]
    end

    @testset "shuffled" begin
        # does not reshuffle on iteration
        shuffled = eachobs(shuffleobs(1:50))
        @test collect(shuffled) == collect(shuffled)

        # does reshuffle
        reshuffled = eachobs(1:50, shuffle = true)
        @test collect(reshuffled) != collect(reshuffled)

        reshuffled = eachobs(1:50, shuffle = true, buffer = true)
        @test collect(reshuffled) != collect(reshuffled)

        reshuffled = eachobs(1:50, shuffle = true, parallel = true)
        @test collect(reshuffled) != collect(reshuffled)

        reshuffled = eachobs(1:50, shuffle = true, buffer = true, parallel = true)
        @test collect(reshuffled) != collect(reshuffled)
    end
    @testset "Argument combinations" begin
        for batchsize ∈ (-1, 2), buffer ∈ (true, false), collate ∈ (nothing, true, false),
                parallel ∈ (true, false), shuffle ∈ (true, false), partial ∈ (true, false)
            if !(buffer isa Bool) && batchsize > 0
                buffer = getobs(BatchView(X; batchsize), 1)
            end
            iter = eachobs(X; batchsize, shuffle, buffer, parallel, partial)
            @test_nowarn for _ in iter end
        end
    end

    @testset "parallel worker count" begin
        # `parallel` accepts a worker count, with `true` = nthreads and `0`/`false` = serial
        @test MLUtils._nworkers(false) == 0
        @test MLUtils._nworkers(0) == 0
        @test MLUtils._nworkers(-3) == 0
        @test MLUtils._nworkers(true) == Threads.nthreads()
        @test MLUtils._nworkers(3) == 3
        # `Bool <: Integer`, so `true` must not be conflated with `1`
        @test MLUtils._nworkers(1) == 1

        # codepath dispatch: 0/false serial, positive ints / true parallel
        mode(d) = typeof(d).parameters[3]
        @test mode(DataLoader(1:50; parallel=false)) === :serial
        @test mode(DataLoader(1:50; parallel=0)) === :serial
        @test mode(DataLoader(1:50; parallel=true)) === :parallel
        @test mode(DataLoader(1:50; parallel=1)) === :parallel
        @test mode(DataLoader(1:50; parallel=4)) === :parallel

        # the raw value round-trips (faithful printing / reconstruction)
        @test DataLoader(1:50; parallel=3).parallel === 3
        @test DataLoader(1:50; parallel=true).parallel === true
        @test DataLoader(1:50; parallel=false).parallel === false

        # every observation is loaded exactly once regardless of worker count
        ref = collect(1:50)
        for p in (false, 0, 1, 2, true)
            @test sort(collect(eachobs(1:50; parallel=p))) == ref
            @test sort(collect(eachobs(1:50; parallel=p, buffer=true))) == ref
            batches = collect(eachobs(1:50; parallel=p, batchsize=5))
            @test length(batches) == 10
            @test sort(reduce(vcat, batches)) == ref
        end
    end
end

@testset "Empty handling" begin
    xtrain = Matrix{Float64}(undef, 3, 0)
    data = DataLoader(xtrain, batchsize = 0, partial = true);
end