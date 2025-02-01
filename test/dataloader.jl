
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

    @testset "Transducers foldl" begin
        dloader = DataLoader(1:10)
        @test foldl(+, Map(x -> x[1]), dloader; init = 0) == 55
        @inferred foldl(+, Map(x -> x[1]), dloader; init = 0)

        dloader = DataLoader(1:10; shuffle = true)
        @test foldl(+, Map(x -> x[1]), dloader; init = 0) == 55

        dloader = DataLoader(1:10; batchsize = 2)
        @test foldl(+, Map(x -> x[1]), dloader; init = 0) == 25

        dloader = DataLoader(1:1000; shuffle = false)
        @test copy(Map(x -> x[1]), Vector{Int}, dloader) == collect(1:1000)

        dloader = DataLoader(1:1000; shuffle = true)
        @test copy(Map(x -> x[1]), Vector{Int}, dloader) != collect(1:1000)

        dloader = DataLoader(1:1000; batchsize = 2, shuffle = false)
        @test copy(Map(x -> x[1]), Vector{Int}, dloader) == collect(1:2:1000)

        dloader = DataLoader(1:1000; batchsize = 2, shuffle = true)
        @test copy(Map(x -> x[1]), Vector{Int}, dloader) != collect(1:2:1000)
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
            
            @test contains(repr(d), "DataLoader(::Tuple{Matrix")
            @test contains(repr(d), "batchsize=3")

            @test contains(repr(MIME"text/plain"(), d), "2-element DataLoader")
            @test contains(repr(MIME"text/plain"(), d), "2×3 Matrix{Float32}, 3-element Vector")
            
            d2 = DataLoader((x = X2, y = Y2), batchsize=2, partial=false)

            @test contains(repr(d2), "DataLoader(::@NamedTuple")
            @test contains(repr(d2), "partial=false")

            @test contains(repr(MIME"text/plain"(), d2), "2-element DataLoader(::@NamedTuple")
            @test contains(repr(MIME"text/plain"(), d2), "x = 2×2 Matrix{Float32}, y = 2-element Vector")
        end
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
end
