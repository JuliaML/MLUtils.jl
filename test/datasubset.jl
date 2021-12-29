@testset "DataSubset constructor" begin
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(9)))
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(9)), 1:2)
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(4,9,10), rand(9)))

    @testset "bounds check" begin
        for var in (vars..., tuples..., CustomType())
            @test_throws BoundsError DataSubset(var, -1:100)
            @test_throws BoundsError DataSubset(var, 1:16)
            @test_throws BoundsError DataSubset(var, [1, 10, 0, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, -10, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, 18, 3])
        end
    end

    @testset "Tuples" begin
        @test typeof(@inferred(DataSubset((X,X)))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), 1:15))) <: DataSubset
        d = DataSubset((X, y), 5:10)
        @test @inferred(getobs(d , 2)) == getobs((X, y), 6)
        @test numobs(d) == 6
    end

    @testset "Array, SubArray, SparseArray" begin
        for var in (Xs, ys, vars...)
            subset = @inferred(DataSubset(var))
            @test subset.data === var
            @test subset.indices === 1:15
            @test typeof(subset) <: DataSubset
            @test @inferred(numobs(subset)) === numobs(var)
            @test @inferred(getobs(subset)) == getobs(var)
            @test @inferred(DataSubset(subset)) === subset
            @test @inferred(DataSubset(subset, 1:15)) === subset
            @test subset[end] == datasubset(var, 15)
            @test @inferred(subset[15]) == datasubset(var, 15)
            @test @inferred(subset[2:5]) == datasubset(var, 2:5)
            for idx in (1:10, [1,10,15,3], [2])
                @test DataSubset(var)[idx] == datasubset(var, idx)
                @test DataSubset(var)[idx] == datasubset(var, collect(idx))
                subset = @inferred(DataSubset(var, idx))
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test @inferred(numobs(subset)) === length(idx)
                @test @inferred(getobs(subset)) == getobs(var, idx)
                @test @inferred(DataSubset(subset)) === subset
                @test @inferred(subset[1]) == datasubset(var, idx[1])
                @test numobs(subset[1:1]) == numobs(DataSubset(var, view(idx, 1:1)))
            end
        end
    end

    @testset "custom types" begin
        @test_throws MethodError DataSubset(EmptyType())
        @test_throws MethodError DataSubset(EmptyType(), 1:10)
        @test_throws BoundsError getobs(DataSubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(DataSubset(CustomType()))) <: DataSubset
        @test numobs(DataSubset(CustomType())) === 15
        @test numobs(DataSubset(CustomType(), 1:10)) === 10
        @test getobs(DataSubset(CustomType())) == collect(1:15)
        @test getobs(DataSubset(CustomType(),1:10),10) == 10
        @test getobs(DataSubset(CustomType(),1:10),[3,5]) == [3,5]
    end
end

@testset "DataSubset getindex and getobs" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = @inferred(DataSubset(var, 5:12))
            @test typeof(@inferred(getobs(subset))) <: Array{Float64,2}
            @test @inferred(numobs(subset)) == length(subset) == 8
            @test @inferred(subset[2:5]) == datasubset(X, 6:9)
            @test @inferred(subset[3:6]) != datasubset(X, 6:9)
            @test @inferred(getobs(subset, 2:5)) == X[:, 6:9]
            @test @inferred(getobs(subset, [3,1,4])) == X[:, [7,5,8]]
            @test typeof(subset[2:5]) <: SubArray
            @test @inferred(subset[collect(2:5)]) == datasubset(X, collect(6:9))
            @test typeof(subset[collect(2:5)]) <: SubArray
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == X[:, 5:12]
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = @inferred(DataSubset(var, 6:10))
            @test typeof(getobs(subset)) <: Array{String,1}
            @test @inferred(numobs(subset)) == length(subset) == 5
            @test @inferred(subset[2:3]) == datasubset(y, 7:8)
            @test @inferred(getobs(subset, 2:3)) == y[7:8]
            @test @inferred(getobs(subset, [2,1,4])) == y[[7,6,9]]
            @test typeof(subset[2:3]) <: SubArray
            @test @inferred(subset[collect(2:3)]) == datasubset(y, collect(7:8))
            @test typeof(subset[collect(2:3)]) <: SubArray
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == y[6:10]
        end
    end
end

@testset "datasubset" begin
    @testset "Array and SubArray" begin
        @test @inferred(datasubset(X)) == Xv
        @test @inferred(datasubset(X)) == Xv
        @test @inferred(datasubset(X)) == Xv
        @test typeof(datasubset(X)) <: SubArray
        @test @inferred(datasubset(Xv)) === Xv
        @test @inferred(datasubset(XX)) == XX
        @test @inferred(datasubset(XXX)) == XXX
        @test typeof(datasubset(XXX)) <: SubArray
        @test @inferred(datasubset(y)) == y
        @test typeof(datasubset(y)) <: SubArray
        @test @inferred(datasubset(yv)) === yv
        for i in (2, 1:15, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset(X,i))   === view(X,:,i)
            @test @inferred(datasubset(Xv,i))  === view(X,:,i)
            @test @inferred(datasubset(Xv,i))  === view(Xv,:,i)
            @test @inferred(datasubset(XX,i))  === view(XX,:,:,i)
            @test @inferred(datasubset(XXX,i)) === view(XXX,:,:,:,i)
            @test @inferred(datasubset(y,i))   === view(y,i)
            @test @inferred(datasubset(yv,i))  === view(y,i)
            @test @inferred(datasubset(yv,i))  === view(yv,i)
            @test @inferred(datasubset(Y,i))   === view(Y,:,i)
        end
    end

    @testset "Tuple of Array and Subarray" begin
        @test @inferred(datasubset((X,y)))   == (X,y)
        @test @inferred(datasubset((X,yv)))  == (X,yv)
        @test @inferred(datasubset((Xv,y)))  == (Xv,y)
        @test @inferred(datasubset((Xv,yv))) == (Xv,yv)
        @test @inferred(datasubset((X,Y)))   == (X,Y)
        @test @inferred(datasubset((XX,X,y))) == (XX,X,y)
        for i in (1:15, 2:10, [2,5,7], [2,1])
            @test @inferred(datasubset((X,y),i))   === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,y),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((X,yv),i))  === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((Xv,yv),i)) === (view(X,:,i), view(y,i))
            @test @inferred(datasubset((XX,X,y),i)) === (view(XX,:,:,i), view(X,:,i),view(y,i))
            # compare if obs match in tuple
            x1, y1 = getobs(datasubset((X1,Y1), i))
            @test all(x1' .== y1)
            x1, y1, z1 = getobs(datasubset((X1,Y1,X1), i))
            @test all(x1' .== y1)
            @test all(x1 .== z1)
        end
    end

    @testset "custom types" begin
        @test_throws MethodError datasubset(EmptyType())
        @test_throws MethodError datasubset(EmptyType(), 1:10)
        @test_throws BoundsError getobs(datasubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(datasubset(CustomType()))) <: DataSubset
        @test datasubset(CustomType()) == DataSubset(CustomType())
        @test datasubset(CustomType(), 2:11) == DataSubset(CustomType(), 2:11)
        @test numobs(datasubset(CustomType())) === 15
        @test numobs(datasubset(CustomType(), 2:11)) === 10
        @test getobs(datasubset(CustomType())) == collect(1:15)
        @test getobs(datasubset(CustomType(), 2:11), 10) == 11
        @test getobs(datasubset(CustomType(), 2:11), [3,5]) == [4,6]
    end
end

@testset "getobs!" begin
    @test getobs!(nothing, datasubset(y, 1)) == "setosa"

    @testset "DataSubset" begin
        xbuf1 = zeros(4,8)
        s1 = DataSubset(X, 2:9)
        @test @inferred(getobs!(xbuf1,s1)) === xbuf1
        @test xbuf1 == getobs(s1)
        xbuf1 = zeros(4,5)
        s1 = DataSubset(X, 5:12)
        @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
        @test xbuf1 == getobs(s1,2:6) == getobs(X,6:10)

        s3 = DataSubset(Xs, 6:10)
        @test @inferred(getobs!(nothing,s3)) == getobs(Xs,6:10)

        s4 = DataSubset(CustomType(), 6:10)
        @test @inferred(getobs!(nothing,s4)) == getobs(s4)
        s5 = DataSubset(CustomType(), 3:10)
        @test @inferred(getobs!(nothing,s5,2:6)) == getobs(s5,2:6)
    end

    @testset "Tuple with DataSubset" begin
        xbuf = zeros(4,2)
        ybuf = ["foo", "bar"]
        s1 = DataSubset(Xs, 5:9)
        s2 = DataSubset(X, 5:9)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test xbuf == getobs(X,6:7)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
        @test getobs!((nothing,xbuf),(s1,s2), 2:3) == (getobs(Xs,6:7),xbuf)
    end
end
