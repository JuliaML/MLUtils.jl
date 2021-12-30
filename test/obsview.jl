@testset "ObsView constructor" begin
    @test_throws DimensionMismatch ObsView((rand(2,10), rand(9)))
    @test_throws DimensionMismatch ObsView((rand(2,10), rand(9)), 1:2)
    @test_throws DimensionMismatch ObsView((rand(2,10), rand(4,9,10), rand(9)))

    @testset "bounds check" begin
        for var in (vars..., tuples..., CustomType())
            @test_throws BoundsError ObsView(var, -1:100)
            @test_throws BoundsError ObsView(var, 1:16)
            @test_throws BoundsError ObsView(var, [1, 10, 0, 3])
            @test_throws BoundsError ObsView(var, [1, 10, -10, 3])
            @test_throws BoundsError ObsView(var, [1, 10, 18, 3])
        end
    end

    @testset "Tuples" begin
        @test typeof(@inferred(ObsView((X,X)))) <: ObsView
        @test typeof(@inferred(ObsView((X,X), 1:15))) <: ObsView
        d = ObsView((X, y), 5:10)
        @test @inferred(getobs(d , 2)) == getobs((X, y), 6)
        @test numobs(d) == 6
    end

    @testset "Array, SubArray, SparseArray" begin
        for var in (Xs, ys, vars...)
            subset = @inferred(ObsView(var))
            @test subset.data === var
            @test subset.indices === 1:15
            @test typeof(subset) <: ObsView
            @test @inferred(numobs(subset)) === numobs(var)
            @test @inferred(getobs(subset)) == getobs(var)
            @test @inferred(ObsView(subset)) === subset
            @test @inferred(ObsView(subset, 1:15)) === subset
            @test subset[end] == MLUtils.datasubset(var, 15)
            @test @inferred(subset[15]) == MLUtils.datasubset(var, 15)
            @test @inferred(subset[2:5]) == MLUtils.datasubset(var, 2:5)
            for idx in (1:10, [1,10,15,3], [2])
                @test ObsView(var)[idx] == MLUtils.datasubset(var, idx)
                @test ObsView(var)[idx] == MLUtils.datasubset(var, collect(idx))
                subset = @inferred(ObsView(var, idx))
                @test typeof(subset) <: ObsView{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test @inferred(numobs(subset)) === length(idx)
                @test @inferred(getobs(subset)) == getobs(var, idx)
                @test @inferred(ObsView(subset)) === subset
                @test @inferred(subset[1]) == MLUtils.datasubset(var, idx[1])
                @test numobs(subset[1:1]) == numobs(ObsView(var, MLUtils.datasubset(idx, 1:1)))
            end
        end
    end

    @testset "custom types" begin
        @test_throws MethodError ObsView(EmptyType())
        @test_throws MethodError ObsView(EmptyType(), 1:10)
        @test_throws BoundsError getobs(ObsView(CustomType(), 11:20), 11)
        @test typeof(@inferred(ObsView(CustomType()))) <: ObsView
        @test numobs(ObsView(CustomType())) === 15
        @test numobs(ObsView(CustomType(), 1:10)) === 10
        @test getobs(ObsView(CustomType())) == collect(1:15)
        @test getobs(ObsView(CustomType(),1:10),10) == 10
        @test getobs(ObsView(CustomType(),1:10),[3,5]) == [3,5]
    end
end

@testset "ObsView getindex and getobs" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = @inferred(ObsView(var, 5:12))
            @test typeof(@inferred(getobs(subset))) <: Array{Float64,2}
            @test @inferred(numobs(subset)) == length(subset) == 8
            @test @inferred(subset[2:5]) == MLUtils.datasubset(X, 6:9)
            @test @inferred(subset[3:6]) != MLUtils.datasubset(X, 6:9)
            @test @inferred(getobs(subset, 2:5)) == X[:, 6:9]
            @test @inferred(getobs(subset, [3,1,4])) == X[:, [7,5,8]]
            @test typeof(subset[2:5]) <: SubArray
            @test @inferred(subset[collect(2:5)]) == MLUtils.datasubset(X, collect(6:9))
            @test typeof(subset[collect(2:5)]) <: SubArray
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == X[:, 5:12]
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = @inferred(ObsView(var, 6:10))
            @test typeof(getobs(subset)) <: Array{String,1}
            @test @inferred(numobs(subset)) == length(subset) == 5
            @test @inferred(subset[2:3]) == MLUtils.datasubset(y, 7:8)
            @test @inferred(getobs(subset, 2:3)) == y[7:8]
            @test @inferred(getobs(subset, [2,1,4])) == y[[7,6,9]]
            @test typeof(subset[2:3]) <: SubArray
            @test @inferred(subset[collect(2:3)]) == MLUtils.datasubset(y, collect(7:8))
            @test typeof(subset[collect(2:3)]) <: SubArray
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == y[6:10]
        end
    end
end

@testset "obsview" begin
    @testset "Array and SubArray" begin
    #     @test @inferred(obsview(X)) == Xv
    #     @test @inferred(obsview(X)) == Xv
    #     @test @inferred(obsview(X)) == Xv
    #     @test typeof(obsview(X)) <: SubArray
    #     @test @inferred(obsview(Xv)) === Xv
    #     @test @inferred(obsview(XX)) == XX
    #     @test @inferred(obsview(XXX)) == XXX
    #     @test typeof(obsview(XXX)) <: SubArray
    #     @test @inferred(obsview(y)) == y
    #     @test typeof(obsview(y)) <: SubArray
    #     @test @inferred(obsview(yv)) === yv
        for i in (2, 1:15, 2:10, [2,5,7], [2,1])
            @test @inferred(obsview(X)[i])   == view(X,:,i)
            @test @inferred(obsview(Xv)[i])  == view(X,:,i)
            @test @inferred(obsview(Xv)[i])  == view(Xv,:,i)
            @test @inferred(obsview(XX)[i])  == view(XX,:,:,i)
            @test @inferred(obsview(XXX)[i]) == view(XXX,:,:,:,i)
            @test @inferred(obsview(y)[i])   == view(y,i)
            @test @inferred(obsview(yv)[i])  == view(y,i)
            @test @inferred(obsview(yv)[i])  == view(yv,i)
            @test @inferred(obsview(Y)[i])   == view(Y,:,i)
        end
    end

    @testset "Tuple of Array and Subarray" begin
        # @test @inferred(obsview((X,y)))   == (X,y)
        # @test @inferred(obsview((X,yv)))  == (X,yv)
        # @test @inferred(obsview((Xv,y)))  == (Xv,y)
        # @test @inferred(obsview((Xv,yv))) == (Xv,yv)
        # @test @inferred(obsview((X,Y)))   == (X,Y)
        # @test @inferred(obsview((XX,X,y))) == (XX,X,y)
        for i in (1:15, 2:10, [2,5,7], [2,1])
            @test @inferred(obsview((X,y))[i])   == (view(X,:,i), view(y,i))
            @test @inferred(obsview((Xv,y))[i])  == (view(X,:,i), view(y,i))
            @test @inferred(obsview((X,yv))[i])  == (view(X,:,i), view(y,i))
            @test @inferred(obsview((Xv,yv))[i]) == (view(X,:,i), view(y,i))
            @test @inferred(obsview((XX,X,y))[i]) == (view(XX,:,:,i), view(X,:,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(obsview((X1,Y1))[i])
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(obsview((X1,Y1,X1))[i])
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    @testset "custom types" begin
        @test_throws MethodError obsview(EmptyType())
        @test_throws MethodError obsview(EmptyType(), 1:10)
        @test_throws BoundsError getobs(obsview(CustomType(), 11:20), 11)
        @test typeof(@inferred(obsview(CustomType()))) <: ObsView
        @test obsview(CustomType()) == ObsView(CustomType())
        @test obsview(CustomType(), 2:11) == ObsView(CustomType(), 2:11)
        @test numobs(obsview(CustomType())) === 15
        @test numobs(obsview(CustomType(), 2:11)) === 10
        @test getobs(obsview(CustomType())) == collect(1:15)
        @test getobs(obsview(CustomType(), 2:11), 10) == 11
        @test getobs(obsview(CustomType(), 2:11), [3,5]) == [4,6]
    end
end

@testset "getobs!" begin
    @test getobs!(nothing, obsview(y, 1)) == "setosa"

    @testset "ObsView" begin
        xbuf1 = zeros(4,8)
        s1 = ObsView(X, 2:9)
        @test @inferred(getobs!(xbuf1,s1)) === xbuf1
        @test xbuf1 == getobs(s1)
        xbuf1 = zeros(4,5)
        s1 = ObsView(X, 5:12)
        @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
        @test xbuf1 == getobs(s1,2:6) == getobs(X,6:10)

        s3 = ObsView(Xs, 6:10)
        @test @inferred(getobs!(nothing,s3)) == getobs(Xs,6:10)

        s4 = ObsView(CustomType(), 6:10)
        @test @inferred(getobs!(nothing,s4)) == getobs(s4)
        s5 = ObsView(CustomType(), 3:10)
        @test @inferred(getobs!(nothing,s5,2:6)) == getobs(s5,2:6)
    end

    @testset "Tuple with ObsView" begin
        xbuf = zeros(4,2)
        ybuf = ["foo", "bar"]
        s1 = ObsView(Xs, 5:9)
        s2 = ObsView(X, 5:9)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test xbuf == getobs(X,6:7)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
    end
end
