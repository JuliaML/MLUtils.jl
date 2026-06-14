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

    @testset "shuffle" begin
        rng = MersenneTwister(42)
        folds = collect(kfolds(rng, 1:15, k=5))
        # every observation appears in exactly one validation set
        @test sort(reduce(vcat, [xval for (_, xval) in folds])) == 1:15
        for (xtr, xval) in folds
            @test length(xtr) == 12
            @test length(xval) == 3
            @test sort([xtr; xval]) == 1:15
        end
        # shuffle actually changes the assignment
        @test collect(kfolds(1:15, k=5, shuffle=true)) != collect(kfolds(1:15, k=5))
        # reproducible given the same rng
        @test collect(kfolds(MersenneTwister(1), 1:15, k=3)) ==
              collect(kfolds(MersenneTwister(1), 1:15, k=3))
    end

    @testset "stratified" begin
        y = [fill(0, 10); fill(1, 20)]
        x = collect(1:30)
        k = 5
        seen = Int[]
        for (xtr, xval) in kfolds(x, k=k, stratified=y)
            # class proportions preserved in each validation set
            @test count(==(0), y[xval]) == 2
            @test count(==(1), y[xval]) == 4
            @test length(xval) == 6
            @test length(xtr) == 24
            @test sort([xtr; xval]) == 1:30
            # train and val are disjoint and cover everything
            @test isempty(intersect(xtr, xval))
            append!(seen, xval)
        end
        # each observation used for validation exactly once
        @test sort(seen) == 1:30

        # stratified + shuffle
        rng = MersenneTwister(0)
        for (xtr, xval) in kfolds(rng, x, k=k, stratified=y)
            @test count(==(0), y[xval]) == 2
            @test count(==(1), y[xval]) == 4
            @test sort([xtr; xval]) == 1:30
        end

        # errors
        @test_throws ArgumentError kfolds(x, k=5, stratified=y[1:5])
        ysmall = [fill(0, 3); fill(1, 27)]  # label 0 has only 3 < k=5
        @test_throws ArgumentError kfolds(x, k=5, stratified=ysmall)
    end
end

@testset "leavepout" begin
    @test leavepout(15) == kfolds(15, k=15)
    @test collect(leavepout(1:15)) == collect(kfolds(1:15, 15))
    @test collect(leavepout(1:15, p=5)) == collect(kfolds(1:15, 3))
end

@testset "timeseries_kfolds" begin
    @testset "int" begin
        # matches MLJ's TimeSeriesCV(nfolds=3) on 1:10
        tr, val = timeseries_kfolds(10; k=3)
        @test tr == [1:4, 1:6, 1:8]
        @test val == [5:6, 7:8, 9:10]

        # validation blocks are contiguous, equal-sized, and ordered after train
        for (itr, iv) in zip(tr, val)
            @test last(itr) < first(iv)
        end

        # gap discards observations between train and val
        @test timeseries_kfolds(10; k=3, gap=1)[1] == [1:3, 1:5, 1:7]
        @test timeseries_kfolds(10; k=3, gap=1)[2] == [5:6, 7:8, 9:10]

        # max_train_size turns the expanding window into a sliding one
        @test timeseries_kfolds(10; k=3, max_train_size=2)[1] == [3:4, 5:6, 7:8]

        @test_throws ArgumentError timeseries_kfolds(10; k=1)
        @test_throws ArgumentError timeseries_kfolds(10; k=10)   # k > n-1
        @test_throws ArgumentError timeseries_kfolds(3; k=3)     # fold_size == 0
        @test_throws ArgumentError timeseries_kfolds(10; k=3, gap=10)
        @test_throws ArgumentError timeseries_kfolds(10; k=3, gap=-1)
    end

    @testset "data" begin
        for (xtr, xval) in timeseries_kfolds(1:10; k=3)
            @test maximum(xtr) < minimum(xval)
        end

        for ((xtr, ytr), (xval, yval)) in timeseries_kfolds((1:10, 11:20); k=3)
            @test maximum(xtr) < minimum(xval)
            @test maximum(ytr) < minimum(yval)
        end
    end
end

