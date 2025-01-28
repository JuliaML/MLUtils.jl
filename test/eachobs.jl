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
