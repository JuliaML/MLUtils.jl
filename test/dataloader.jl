
@testset "DataLoader" begin
    X2 = reshape([1:10;], (2, 5))
    Y2 = [1:5;]

    d = DataLoader(X2, batchsize=2)
    # @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == typeof(X2)
    @test length(batches) == 3
    @test batches[1] == X2[:,1:2]
    @test batches[2] == X2[:,3:4]
    @test batches[3] == X2[:,5:5]

    d = DataLoader(X2, batchsize=2, partial=false)
    # @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == typeof(X2)
    @test length(batches) == 2
    @test batches[1] == X2[:,1:2]
    @test batches[2] == X2[:,3:4]

    d = DataLoader((X2,), batchsize=2, partial=false)
    # @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X2)}
    @test length(batches) == 2
    @test batches[1] == (X2[:,1:2],)
    @test batches[2] == (X2[:,3:4],)

    d = DataLoader((X2, Y2), batchsize=2)
    # @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == Tuple{typeof(X2), typeof(Y2)}
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
    # @inferred first(d)
    batches = collect(d)
    @test eltype(batches) == eltype(d) == NamedTuple{(:x, :y), Tuple{typeof(X2), typeof(Y2)}}
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

    @testset "shuffle & rng" begin
    # specify the rng
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
        @test eltype(dloader) == Dict{String, Array{Float64}}
        c = collect(dloader)
        @test c[1] == Dict("x" => data["x"][:,1:2], "y" => data["y"][1:2])
        @test c[2] == Dict("x" => data["x"][:,3:4], "y" => data["y"][3:4])

        data = Dict("x" => rand(2,4), "y" => rand(2,4))
        dloader = DataLoader(data, batchsize=2)
        @test eltype(dloader) == Dict{String, Matrix{Float64}}
    end
end