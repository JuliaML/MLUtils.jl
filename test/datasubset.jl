
# --------------------------------------------------------------------
# create some test data


Random.seed!(1335)
X = rand(4, 150)
y = repeat(["setosa","versicolor","virginica"], inner = 50)
Y = permutedims(hcat(y,y), [2,1])
Yt = hcat(y,y)
yt = Y[1:1,:]
Xv = view(X,:,:)
yv = view(y,:)
XX = rand(20,30,150)
XXX = rand(3,20,30,150)
vars = (X, Xv, yv, XX, XXX, y)
tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
Xs = sprand(10, 150, 0.5)
ys = sprand(150, 0.5)
# to compare if obs match
X1 = hcat((1:150 for i = 1:10)...)'
Y1 = collect(1:150)

struct EmptyType end

struct CustomType end
MLUtils.numobs(::CustomType) = 100
MLUtils.getobs(::CustomType, i::Int) = i
MLUtils.getobs(::CustomType, i::AbstractVector) = collect(i)
# MLUtils.gettargets(::CustomType, i::Int) = "obs $i"
# MLUtils.gettargets(::CustomType, i::AbstractVector) = "batch $i"



@testset "DataSubset constructor" begin
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(9)))
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(9)), 1:2)
    @test_throws DimensionMismatch DataSubset((rand(2,10), rand(4,9,10), rand(9)))

    @testset "bounds check" begin
        for var in (vars..., tuples..., CustomType())
            @test_throws BoundsError DataSubset(var, -1:100)
            @test_throws BoundsError DataSubset(var, 1:151)
            @test_throws BoundsError DataSubset(var, [1, 10, 0, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, -10, 3])
            @test_throws BoundsError DataSubset(var, [1, 10, 180, 3])
        end
    end

    @testset "Tuples" begin
        @test typeof(@inferred(DataSubset((X,X)))) <: DataSubset
        @test typeof(@inferred(DataSubset((X,X), 1:150))) <: DataSubset
        d = DataSubset((X, y), 5:10)
        @test @inferred(getobs(d , 2)) == getobs((X, y), 6)
        @test numobs(d) == 6
    end

    @testset "Array, SubArray, SparseArray" begin
        for var in (Xs, ys, vars...)
            subset = @inferred(DataSubset(var))
            @test subset.data === var
            @test subset.indices === 1:150
            @test typeof(subset) <: DataSubset
            @test @inferred(numobs(subset)) === numobs(var)
            @test @inferred(getobs(subset)) == getobs(var)
            @test @inferred(DataSubset(subset)) === subset
            @test @inferred(DataSubset(subset, 1:150)) === subset
            @test subset[end] == DataSubset(var, 150)
            @test @inferred(subset[150]) == DataSubset(var, 150)
            @test @inferred(subset[20:25]) == DataSubset(var, 20:25)
            for idx in (1:100, [1,10,150,3], [2])
                @test DataSubset(var)[idx] == DataSubset(var, idx)
                @test DataSubset(var)[idx] == DataSubset(var, collect(idx))
                subset = @inferred(DataSubset(var, idx))
                @test typeof(subset) <: DataSubset{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test @inferred(numobs(subset)) === length(idx)
                @test @inferred(getobs(subset)) == getobs(var, idx)
                @test @inferred(DataSubset(subset)) === subset
                @test @inferred(subset[1]) == DataSubset(var, idx[1])
                if typeof(idx) <: AbstractRange
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, idx[1:1]))
                    @test numobs(subset[1:1]) == numobs(DataSubset(var, idx[1:1]))
                else
                    @test typeof(@inferred(subset[1:1])) == typeof(DataSubset(var, view(idx, 1:1)))
                    @test numobs(subset[1:1]) == numobs(DataSubset(var, view(idx, 1:1)))
                end
            end
        end
    end

    @testset "custom types" begin
        @test_throws MethodError DataSubset(EmptyType())
        @test_throws MethodError DataSubset(EmptyType(), 1:10)
        @test_throws BoundsError getobs(DataSubset(CustomType(), 11:20), 11)
        @test typeof(@inferred(DataSubset(CustomType()))) <: DataSubset
        @test numobs(DataSubset(CustomType())) === 100
        @test numobs(DataSubset(CustomType(), 11:20)) === 10
        @test getobs(DataSubset(CustomType())) == collect(1:100)
        @test getobs(DataSubset(CustomType(),11:20),10) == 20
        @test getobs(DataSubset(CustomType(),11:20),[3,5]) == [13,15]
    end
end

@testset "DataSubset getindex and getobs" begin
    @testset "Matrix and SubArray{T,2}" begin
        for var in (X, Xv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(@inferred(getobs(subset))) <: Array{Float64,2}
            @test @inferred(numobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(X, 110:120)
            @test @inferred(subset[11:21]) != DataSubset(X, 110:120)
            @test @inferred(getobs(subset, 10:20)) == X[:, 110:120]
            @test @inferred(getobs(subset, [11,10,14])) == X[:, [111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(X, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == X[:, 101:150]
        end
    end

    @testset "Vector and SubArray{T,1}" begin
        for var in (y, yv)
            subset = @inferred(DataSubset(var, 101:150))
            @test typeof(getobs(subset)) <: Array{String,1}
            @test @inferred(numobs(subset)) == length(subset) == 50
            @test @inferred(subset[10:20]) == DataSubset(y, 110:120)
            @test @inferred(getobs(subset, 10:20)) == y[110:120]
            @test @inferred(getobs(subset, [11,10,14])) == y[[111,110,114]]
            @test typeof(subset[10:20]) <: DataSubset
            @test @inferred(subset[collect(10:20)]) == DataSubset(y, collect(110:120))
            @test typeof(subset[collect(10:20)]) <: DataSubset
            @test @inferred(getobs(subset)) == getobs(subset[1:end]) == y[101:150]
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
        for i in (2, 1:150, 2:10, [2,5,7], [2,1])
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
        for i in (1:150, 2:10, [2,5,7], [2,1])
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
        @test datasubset(CustomType(), 11:20) == DataSubset(CustomType(), 11:20)
        @test numobs(datasubset(CustomType())) === 100
        @test numobs(datasubset(CustomType(), 11:20)) === 10
        @test getobs(datasubset(CustomType())) == collect(1:100)
        @test getobs(datasubset(CustomType(), 11:20), 10) == 20
        @test getobs(datasubset(CustomType(), 11:20), [3,5]) == [13,15]
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
        s1 = DataSubset(X, 10:17)
        @test @inferred(getobs!(xbuf1,s1,2:6)) === xbuf1
        @test xbuf1 == getobs(s1,2:6) == getobs(X,11:15)

        s3 = DataSubset(Xs, 11:15)
        @test @inferred(getobs!(nothing,s3)) == getobs(Xs,11:15)

        s4 = DataSubset(CustomType(), 6:10)
        @test @inferred(getobs!(nothing,s4)) == getobs(s4)
        s5 = DataSubset(CustomType(), 9:20)
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
