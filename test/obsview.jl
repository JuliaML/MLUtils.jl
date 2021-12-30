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
            @test subset[end] == datasubset(var, 15)
            @test @inferred(subset[15]) == datasubset(var, 15)
            @test @inferred(subset[2:5]) == datasubset(var, 2:5)
            for idx in (1:10, [1,10,15,3], [2])
                @test ObsView(var)[idx] == datasubset(var, idx)
                @test ObsView(var)[idx] == datasubset(var, collect(idx))
                subset = @inferred(ObsView(var, idx))
                @test typeof(subset) <: ObsView{typeof(var), typeof(idx)}
                @test subset.data === var
                @test subset.indices === idx
                @test @inferred(numobs(subset)) === length(idx)
                @test @inferred(getobs(subset)) == getobs(var, idx)
                @test @inferred(ObsView(subset)) === subset
                @test @inferred(subset[1]) == datasubset(var, idx[1])
                @test numobs(subset[1:1]) == numobs(ObsView(var, datasubset(idx, 1:1)))
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
            subset = @inferred(ObsView(var, 6:10))
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

@testset "ObsView other" begin
    # @test ObsView <: AbstractVector
    # @test ObsView <: DataView
    # @test ObsView <: AbstractObsView
    # @test ObsView <: AbstractObsIterator
    # @test ObsView <: AbstractDataIterator
    @test obsview === ObsView

    @testset "constructor" begin
        # @test_throws DimensionMismatch ObsView((rand(2,10),rand(9)))
        # @test_throws DimensionMismatch ObsView((rand(2,10),rand(4,9,10),rand(9)))
        @test_throws MethodError ObsView(EmptyType())
        @test_throws MethodError ObsView((EmptyType(),EmptyType()))
        @test_throws MethodError ObsView((EmptyType(),EmptyType()))
        for var in (vars..., Xs, ys)
            A = @inferred(ObsView(var))
        end
        for var in (vars..., tuples..., Xs, ys)
            A = ObsView(var)
            @test @inferred(parent(A)) === var
            @test @inferred(ObsView(A)) == A
            @test @inferred(ObsView(var)) == A
        end
    end

    @testset "typestability" begin
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(ObsView(var))) <: ObsView
            @test typeof(@inferred(ObsView(var))) <: ObsView
        end
        for tup in tuples
            @test typeof(@inferred(ObsView(tup))) <: ObsView
        end
        @test typeof(@inferred(ObsView(CustomType()))) <: ObsView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            # for var in (vars[1], )
            A = ObsView(var)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[16]
            @test @inferred(numobs(A)) == 15
            @test @inferred(length(A)) == 15
            @test @inferred(size(A)) == (15,)
            @test @inferred(A[2:3]) == datasubset(var, 2:3)
            @test @inferred(A[[1,3]]) == datasubset(var, [1,3])
            @test @inferred(A[1]) == datasubset(var, 1)
            @test @inferred(A[11]) == datasubset(var, 11)
            @test @inferred(A[15]) == datasubset(var, 15)
            @test A[end] == A[15]
            @test @inferred(getobs(A,1)) == getobs(var, 1)
            @test @inferred(getobs(A,11)) == getobs(var, 11)
            @test @inferred(getobs(A,15)) == getobs(var, 15)
            @test typeof(@inferred(collect(A))) <: Vector
        end
        for var in vars
            A = ObsView(var)
            @test getobs(A) == var
            @test getobs.(A) == [getobs(var,i) for i in 1:numobs(var)]
        end
        for var in tuples
            A = ObsView(var)
            @test getobs(A) == var
            @test getobs.(A) == [map(x->getobs(x,i), var) for i in 1:numobs(var)]
        end
    end

    @testset "subsetting" begin
        for var_raw in (vars..., tuples..., Xs, ys)
            for var in (var_raw, ObsView(var_raw))
                A = ObsView(var)
                @test getobs(@inferred(obsview(A))) == @inferred(getobs(A))
    
                S = @inferred(obsview(A, 1:5))
                @test typeof(S) <: ObsView
                @test @inferred(length(S)) == 5
                @test @inferred(size(S)) == (5,)
                @test @inferred(A[1:5]) == S[:]
                @test @inferred(getobs(A,1:5)) == getobs(S)
                @test @inferred(getobs(S)) == getobs(ObsView(obsview(var,1:5)))
    
                S = @inferred(ObsView(A, 1:5))
                @test typeof(S) <: ObsView
            end
        end
        A = ObsView(X)
        @test typeof(A.data) <: Array
        S = @inferred(obsview(A))
        @test typeof(S) <: ObsView
        @test @inferred(length(S)) == 15
        @test typeof(S.data) <: Array
    end

    @testset "iteration" begin
        count = 0
        for (i,x) in enumerate(ObsView(X1))
            @test all(i .== x)
            count += 1
        end
        @test count == 15
    end
end
