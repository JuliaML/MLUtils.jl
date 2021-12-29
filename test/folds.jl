@testset "kfolds" begin
    @test collect(kfolds(1:15, k=5)) == collect(kfolds(1:15, 5))
    @test kfolds(15, k=5) == kfolds(15, 5)

    @testset "int" begin
        itrain, ival = kfolds(15, k=5)
        @test size(itrain) == (5,)
        @test size(ival) == (5,)
        @test all(i -> size(i) == (12,), itrain)
        @test all(i -> size(i) == (3,), ival)
        tot = [[it;iv] for (it, iv) in zip(itrain, ival)]
        @test all(x -> sort(x) == 1:15, tot)
    end

    @testset "data" begin
        for (xtr, xval) in kfolds(1:15, k=5)
            @test size(xtr) == (12,)
            @test size(xval) == (3,)
            @test sort([xtr; xval]) == 1:15
        end

        for ((xtr,ytr), (xval, yval)) in kfolds((1:15, 11:25), k=5)
            @test size(xtr) == (12,)
            @test size(xval) == (3,)
            @test size(ytr) == (12,)
            @test size(yval) == (3,)
            @test sort([xtr; xval]) == 1:15
            @test sort([ytr; yval]) == 11:25
        end
    end
end

@testset "leavepout" begin
    @test leavepout(15) == kfolds(15, k=15)
    @test collect(leavepout(1:15)) == collect(kfolds(1:15, 15))
    @test collect(leavepout(1:15, p=5)) == collect(kfolds(1:15, 3))
end

