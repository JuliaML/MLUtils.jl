@testset "mapobs" begin
    data = 1:10
    mdata = mapobs(-, data)
    @test getobs(mdata, 8) == -8

    mdata2 = mapobs((-, x -> 2x), data)
    @test getobs(mdata2, 8) == (-8, 16)

    nameddata = mapobs((x = sqrt, y = log), data)
    @test getobs(nameddata, 10) == (x = sqrt(10), y = log(10))
    @test getobs(nameddata.x, 10) == sqrt(10)

    # multiple obs and indexing
    mdata = mapobs(x -> sum(x.a) + sum(x.b), (a = 1:10, b = 11:20))
    @test mdata[1:2] == 26
end

@testset "filterobs" begin
    data = 1:10
    fdata = filterobs(>(5), data)
    @test numobs(fdata) == 5
end

@testset "groupobs" begin
    data = -10:10
    datas = groupobs(>(0), data)
    @test length(datas) == 2
end

@testset "joinobs" begin
    data1, data2 = 1:10, 11:20
    jdata = joinobs(data1, data2)
    @test getobs(jdata, 15) == 15
end

@testset "shuffleobs" begin
    @test_throws DimensionMismatch shuffleobs((X, rand(149)))

    @testset "typestability" begin
        for var in vars
            @test typeof(@inferred(shuffleobs(var))) <: SubArray
        end
        for tup in tuples
            @test typeof(@inferred(shuffleobs(tup))) <: Tuple
        end
    end

    @testset "Array and SubArray" begin
        for var in vars
            @test size(shuffleobs(var)) == size(var)
        end
        # tests if all obs are still present and none duplicated
        @test sum(shuffleobs(Y1)) == 120
    end

    @testset "Tuple of Array and SubArray" begin
        for var in ((X,yv), (Xv,y), tuples...)
            @test_throws MethodError shuffleobs(var...)
            @test typeof(shuffleobs(var)) <: Tuple
            @test all(map(x->(typeof(x)<:SubArray), shuffleobs(var)))
            @test all(map(x->(numobs(x)===15), shuffleobs(var)))
        end
        # tests if all obs are still present and none duplicated
        # also tests that both paramter are shuffled identically
        x1, y1, z1 = shuffleobs((X1,Y1,X1))
        @test vec(sum(x1,dims=2)) == fill(120,10)
        @test vec(sum(z1,dims=2)) == fill(120,10)
        @test sum(y1) == 120
        @test all(x1' .== y1)
        @test all(z1' .== y1)
    end

    @testset "SparseArray" begin
        for var in (Xs, ys)
            @test typeof(shuffleobs(var)) <: SubArray
            @test numobs(shuffleobs(var)) == numobs(var)
        end
        # tests if all obs are still present and none duplicated
        @test vec(sum(getobs(shuffleobs(sparse(X1))),dims=2)) == fill(120,10)
        @test sum(getobs(shuffleobs(sparse(Y1)))) == 120
    end

    @testset "Tuple of SparseArray" begin
        for var in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys))
            @test_throws MethodError shuffleobs(var...)
            @test typeof(shuffleobs(var)) <: Tuple
            @test numobs(shuffleobs(var)) == numobs(var)
        end
        # tests if all obs are still present and none duplicated
        # also tests that both paramter are shuffled identically
        x1, y1 = getobs(shuffleobs((sparse(X1),sparse(Y1))))
        @test vec(sum(x1,dims=2)) == fill(120,10)
        @test sum(y1) == 120
        @test all(x1' .== y1)
    end

    @testset "RNG" begin
        # tests reproducibility
        explicit_shuffle = shuffleobs(MersenneTwister(42), (X, y))
        @test explicit_shuffle == shuffleobs(MersenneTwister(42), (X, y))
    end

end
