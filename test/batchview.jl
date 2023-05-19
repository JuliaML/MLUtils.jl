using MLUtils: obsview

@testset "BatchView" begin

    @testset "constructor" begin
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(9)))
        @test_throws DimensionMismatch BatchView((rand(2,10),rand(4,9,10),rand(9)))
        @test_throws MethodError BatchView(EmptyType())
        for var in (vars..., tuples..., Xs, ys)
            @test_throws MethodError BatchView(var...)
            @test_throws MethodError BatchView(var, 16)

            A = BatchView(var, batchsize=3)
            @test length(A) == 5
            @test batchsize(A) == 3
            @test numobs(A) == length(A)
            @test @inferred(parent(A)) === var
        end
        A = BatchView(X, batchsize=16)
        @test length(A) == 1
        @test batchsize(A) == 15
    end

    @testset "collated" begin
        @test BatchView(X, batchsize=2, collate=true)[1] |> size == (4, 2)
        @test BatchView(X, batchsize=2, collate=false)[1] |> size == (2,)
        @test size.(BatchView(tuples[1], batchsize=2, collate=true)[1]) == ((4, 2), (2,))
        @test BatchView(tuples[1], batchsize=2, collate=false)[1] |> size == (2,)
    end

    @testset "typestability" begin
        for var in (vars..., tuples..., Xs, ys), batchsize in (1, 3), partial in (true, false), collate in (Val(true), Val(false), Val(nothing))
            @test typeof(@inferred(BatchView(var; batchsize, partial, collate))) <: BatchView
        end
        @test typeof(@inferred(BatchView(CustomType()))) <: BatchView
    end

    @testset "AbstractArray interface" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var, batchsize=5)
            @test_throws BoundsError A[-1]
            @test_throws BoundsError A[4]
            @test @inferred(numobs(A)) == 3
            @test @inferred(length(A)) == 3
            @test @inferred(batchsize(A)) == 5
            @test @inferred(size(A)) == (3,)
            @test @inferred(getobs(A[2:3])) == getobs(BatchView(ObsView(var, 6:15), batchsize=5))
            @test @inferred(getobs(A[[1,3]])) == getobs(BatchView(ObsView(var, [1:5..., 11:15...]), batchsize=5))
            @test @inferred(A[1]) == obsview(var, 1:5)
            @test @inferred(A[2]) == obsview(var, 6:10)
            @test @inferred(A[3]) == obsview(var, 11:15)
            @test A[end] == A[3]
            @test @inferred(getobs(A,1)) == getobs(var, 1:5)
            @test @inferred(getobs(A,2)) == getobs(var, 6:10)
            @test @inferred(getobs(A,3)) == getobs(var, 11:15)
            @test typeof(@inferred(collect(A))) <: Vector
        end
        for var in (vars..., tuples...)
            A = BatchView(var, batchsize=5)
            @test @inferred(getobs(A)) == var
            @test A[2:3] == obsview(var, [6:15;])
            @test A[[1,3]] == obsview(var, [1:5..., 11:15...])
        end
    end


    @testset "subsetting" begin
        for var in (vars..., tuples..., Xs, ys)
            A = BatchView(var, batchsize=3)
            @test getobs(@inferred(ObsView(A))) == @inferred(getobs(A))
            @test_throws BoundsError ObsView(A,1:6)

            S = @inferred(ObsView(A, 1:2))
            @test typeof(S) <: ObsView
            @test @inferred(numobs(S)) == 2
            @test @inferred(length(S)) == numobs(S)
            @test @inferred(size(S)) == (length(S),)
            @test getobs(@inferred(A[1:2])) == getobs(S)
            @test @inferred(getobs(A,1:2)) == getobs(S)
            @test @inferred(getobs(S)) == getobs(BatchView(ObsView(var,1:6),batchsize=3))

            S = @inferred(ObsView(A, 1:2))
            @test typeof(S) <: ObsView
        end
        A = BatchView(X, batchsize=3)
        @test typeof(A.data) <: Array
        S = @inferred(obsview(A))
        S === A
    end

    @testset "obsview and batchview" begin
        x = rand(2, 6)
        bv = BatchView(x; batchsize=2)
        ov = obsview(bv, 1:2)
        @test getobs(ov, 1) == x[:,1:2]
    end

    # @testset "nesting with ObsView" begin
    #     for var in vars
    #         @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Union{SubArray,String}
    #     end
    #     for var in tuples
    #         @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: Tuple
    #     end
    #     for var in (Xs, ys)
    #         @test eltype(@inferred(BatchView(ObsView(var)))[1]) <: SubArray
    #     end
    # end

    @testset "partial=false" begin
        x = [1:12;]
        bv = BatchView(x, batchsize=5, partial=false)
        @test length(bv) == 2
        @test bv[1] == 1:5
        @test bv[2] == 6:10
        @test_throws BoundsError bv[3]
    end


    @testset "getobs!" begin
        buf1 = rand(4, 3)
        bv = BatchView(X, batchsize=3) 
        @test @inferred(getobs!(buf1, bv, 2)) === buf1
        @test buf1 == getobs(bv, 2)
        
        buf2 = rand(4, 6)
        @test @inferred(getobs!(buf2, bv, [1,3])) === buf2
        @test buf2 == getobs(bv, [1,3])

        @testset "custom type" begin # issue #156
            struct DummyData{X} 
                x::X
            end
            MLUtils.numobs(data::DummyData) = numobs(data.x)
            MLUtils.getobs(data::DummyData, idx) = getobs(data.x, idx)
            MLUtils.getobs!(buffer, data::DummyData, idx) = getobs!(buffer, data.x, idx)
            
            data = DummyData(X)
            buf = rand(4, 3)
            bv = BatchView(data, batchsize=3)
            @test @inferred(getobs!(buf, bv, 2)) === buf
            @test buf == getobs(bv, 2)
        end
    end
end
