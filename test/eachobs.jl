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
end
