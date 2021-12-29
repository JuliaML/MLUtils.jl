@testset "ObsView" begin
    @test ObsView <: AbstractVector
    @test ObsView <: DataView
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
            A = ObsView(var)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[16]
            @test @inferred(numobs(A)) == 15
            @test @inferred(length(A)) == 15
            @test @inferred(size(A)) == (15,)
            @test @inferred(A[2:3]) == ObsView(datasubset(var, 2:3))
            @test @inferred(A[[1,3]]) == ObsView(datasubset(var, [1,3]))
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
            @test @inferred(getobs(A)) == [getobs(var,i) for i in 1:numobs(var)]
        end
        for var in tuples
            A = ObsView(var)
            @test @inferred(getobs(A)) == [map(x->getobs(x,i), var) for i in 1:numobs(var)]
        end
    end

    @testset "subsetting" begin
        for var_raw in (vars..., tuples..., Xs, ys)
            for var in (var_raw, DataSubset(var_raw))
                A = ObsView(var)
                @test getobs(@inferred(datasubset(A))) == @inferred(getobs(A))
    
                S = @inferred(datasubset(A, 1:5))
                @test typeof(S) <: ObsView
                @test @inferred(length(S)) == 5
                @test @inferred(size(S)) == (5,)
                @test @inferred(A[1:5]) == S
                @test @inferred(getobs(A,1:5)) == getobs(S)
                @test @inferred(getobs(S)) == getobs(ObsView(datasubset(var,1:5)))
    
                S = @inferred(DataSubset(A, 1:5))
                @test typeof(S) <: DataSubset
            end
        end
        A = ObsView(X)
        @test typeof(A.data) <: Array
        S = @inferred(datasubset(A))
        @test typeof(S) <: ObsView
        @test @inferred(length(S)) == 15
        @test typeof(S.data) <: SubArray
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

# @testset "_compute_batch_settings" begin
#     @test MLUtils._compute_batch_settings(X) === (1,15)
#     @test MLUtils._compute_batch_settings(Xv) === (1,15)
#     @test MLUtils._compute_batch_settings(Xs) === (1,15)
#     @test MLUtils._compute_batch_settings(DataSubset(X)) === (1,15)
#     @test MLUtils._compute_batch_settings((X,y)) === (1,15)
#     @test MLUtils._compute_batch_settings((Xv,yv)) === (1,15)

#     @test_throws ArgumentError MLUtils._compute_batch_settings(X, 160)
#     @test_throws ArgumentError MLUtils._compute_batch_settings(X, 1, 160)
#     @test_throws ArgumentError MLUtils._compute_batch_settings(X, 10, 20)

#     for inner in (Xs, ys, vars...), var in (inner, DataSubset(inner))
#         @test MLUtils._compute_batch_settings(var,3) === (3,5)
#         @test MLUtils._compute_batch_settings(var,0,3) === (5,3)
#         @test MLUtils._compute_batch_settings(var,-1,3) === (5,3)
#         @test MLUtils._compute_batch_settings(var,3,3) === (3,5)
#         @test MLUtils._compute_batch_settings(var,5,1) === (5,3)
#         @test MLUtils._compute_batch_settings(var,5) === (5,3)
#         @test MLUtils._compute_batch_settings(var,0,5) === (3,5)
#         @test MLUtils._compute_batch_settings(var,-1,5) === (3,5)
#     end
# end

@testset "BatchView" begin
    @test BatchView <: AbstractVector
    @test BatchView <: DataView
    @test batchview == BatchView
    # @test_throws MethodError oversample(BatchView(X))
    # @test_throws MethodError undersample(BatchView(X))
    # @test_throws MethodError stratifiedobs(BatchView(X))

    @testset "constructor" begin
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(4,9,10),rand(9)))
        @test_throws MethodError BatchView(EmptyType())
        for var in (vars..., tuples..., Xs, ys)
            @test_throws MethodError BatchView(var...)
            @test_throws MethodError BatchView(var, 16)
            
            A = BatchView(var, size=3)
            @test length(A) == 5
            @test batchsize(A) == 3
            @test numobs(A) == length(A)
            @test @inferred(parent(A)) === var
        end
        A = BatchView(X, size=16)
        @test length(A) == 1
        @test batchsize(A) == 15            
    end


    @testset "typestability" begin
        for var in (vars..., tuples..., Xs, ys)
            @test typeof(@inferred(BatchView(var))) <: BatchView
            @test typeof(@inferred(BatchView(var, size=3))) <: BatchView
            @test typeof(@inferred(BatchView(var, size=3, partial=true))) <: BatchView
            @test typeof(@inferred(BatchView(var, size=3, partial=false))) <: BatchView
        end
        @test typeof(@inferred(BatchView(CustomType()))) <: BatchView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var, size=5)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[4]
            @test @inferred(numobs(A)) == 3
            @test @inferred(length(A)) == 3
            @test @inferred(batchsize(A)) == 5
            @test @inferred(size(A)) == (3,)
            @test @inferred(getobs(A[2:3])) == getobs(BatchView(datasubset(var, 6:15), size=5))
            @test @inferred(getobs(A[[1,3]])) == getobs(BatchView(datasubset(var, [1:5..., 11:15...]), size=5))
            @test @inferred(A[1]) == datasubset(var, 1:5)
            @test @inferred(A[2]) == datasubset(var, 6:10)
            @test @inferred(A[3]) == datasubset(var, 11:15)
            @test A[end] == A[3]
            @test @inferred(getobs(A,1)) == getobs(var, 1:5)
            @test @inferred(getobs(A,2)) == getobs(var, 6:10)
            @test @inferred(getobs(A,3)) == getobs(var, 11:15)
            @test typeof(@inferred(collect(A))) <: Vector
        end
        for var in (vars..., tuples...)
            A = BatchView(var, size=5)
            @test @inferred(getobs(A)) == A
            @test @inferred(A[2:3]) == BatchView(datasubset(var, 6:15), size=5)
            @test @inferred(A[[1,3]]) == BatchView(datasubset(var, [1:5..., 11:15...]), size=5)
        end
    end


    @testset "subsetting" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var, size=3)
            @test getobs(@inferred(datasubset(A))) == @inferred(getobs(A))
            @test_throws BoundsError datasubset(A,1:6)
            
            S = @inferred(datasubset(A, 1:2))
            @test typeof(S) <: BatchView
            @test @inferred(numobs(S)) == 2
            @test @inferred(length(S)) == numobs(S)
            @test @inferred(size(S)) == (length(S),)
            @test getobs(@inferred(A[1:2])) == getobs(S)
            @test @inferred(getobs(A,1:2)) == getobs(S)
            @test @inferred(getobs(S)) == getobs(BatchView(datasubset(var,1:6),size=3))
            
            S = @inferred(DataSubset(A, 1:2))
            @test typeof(S) <: DataSubset
        end
        A = BatchView(X, size=3)
        @test typeof(A.data) <: Array
        S = @inferred(datasubset(A))
        @test typeof(S) <: BatchView
        @test @inferred(length(S)) == 5
        @test typeof(S.data) <: SubArray
    end

    @testset "nesting with ObsView" begin
        for var in vars
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Union{SubArray,String}
        end
        for var in tuples
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Tuple
        end
        for var in (Xs, ys)
            @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: SubArray
        end
    end
end
