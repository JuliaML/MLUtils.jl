@test_throws DimensionMismatch splitobs((X, rand(149)), at=0.7)

@testset "typestability" begin
    @testset "Int" begin
        @test typeof(@inferred(splitobs(10, at=0.))) <: NTuple{2}
        @test eltype(@inferred(splitobs(10, at=0.))) <: UnitRange
        @test typeof(@inferred(splitobs(10, at=1.))) <: NTuple{2}
        @test eltype(@inferred(splitobs(10, at=1.))) <: UnitRange
        @test typeof(@inferred(splitobs(10, at=0.7))) <: NTuple{2}
        @test eltype(@inferred(splitobs(10, at=0.7))) <: UnitRange
        @test typeof(@inferred(splitobs(10, at=0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(10, at=(0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(10, at=0.5))) <: UnitRange
        @test eltype(@inferred(splitobs(10, at=(0.5,0.2)))) <: UnitRange
        @test eltype(@inferred(splitobs(10, at=0.5))) <: UnitRange
        @test eltype(@inferred(splitobs(10, at=(0.,0.2)))) <: UnitRange
    end
    for var in vars
        @test typeof(@inferred(splitobs(var, at=0.7))) <: NTuple{2}
        @test eltype(@inferred(splitobs(var, at=0.7))) <: SubArray
        @test typeof(@inferred(splitobs(var, at=0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(var, at=(0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(var, at=0.5))) <: SubArray
        @test eltype(@inferred(splitobs(var, at=(0.5,0.2)))) <: SubArray
    end
    for tup in tuples
        @test typeof(@inferred(splitobs(tup, at=0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(tup, at=(0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(tup, at=0.5))) <: Tuple
        @test eltype(@inferred(splitobs(tup, at=(0.5,0.2)))) <: Tuple
    end
end

@testset "Int" begin
    @test splitobs(10, at=0.7) == (1:7,8:10)
    @test splitobs(10, at=0.5) == (1:5,6:10)
    @test splitobs(10, at=(0.5,0.3)) == (1:5,6:8,9:10)
    @test splitobs(150, at=0.7) == splitobs(150, at=0.7)
    @test splitobs(150, at=0.5) == splitobs(150, at=0.5)
    @test splitobs(150, at=(0.5,0.2)) == splitobs(150, at=(0.5,0.2))
    @test numobs.(splitobs(150, at=0.7)) == (105,45)
    @test numobs.(splitobs(150, at=(.2,.3))) == (30,45,75)
    @test numobs.(splitobs(150, at=(.1,.2,.3))) == (15,30,45,60)
    # tests if all obs are still present and none duplicated
    @test sum(sum.(getobs.(splitobs(150, at=0.7)))) == 11325
    @test sum(sum.(splitobs(150,at=.1))) == 11325
    @test sum(sum.(splitobs(150,at=(.2,.1)))) == 11325
    @test sum(sum.(splitobs(150,at=(.1,.4,.2)))) == 11325
    @test sum.(splitobs(150, at=0.7)) == (5565, 5760)
end

@testset "Array, SparseArray, and SubArray" begin
    for var in (Xs, ys, vars...)
        @test numobs.(splitobs(var, at=10)) == (10,5)
        @test numobs.(splitobs(var, at=0.7)) == (10, 5)
        @test numobs.(splitobs(var, at=(.2,.3))) == (3,4, 8)
        @test numobs.(splitobs(var, at=(.11,.2,.3))) == (2,3,4,6)
        @test numobs.(splitobs(var, at=(2,3,4))) == (2,3,4,6)
    end
    # tests if all obs are still present and none duplicated
    @test sum(sum.(splitobs(X1,at=.1))) == 120
    @test sum(sum.(splitobs(X1,at=(.2,.1)))) == 120
    @test sum(sum.(splitobs(X1,at=(.1,.4,.2)))) == 120
end

@testset "Tuple of Array, SparseArray, and SubArray" begin
    for tup in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys), (X,yv), (Xv,y), tuples...)
        @test_throws MethodError splitobs(tup..., at=0.5)
        @test all(map(x->(typeof(x)<:Tuple), splitobs(tup,at=0.5)))
        @test numobs.(splitobs(tup, at=0.6)) == (9, 6)
        @test numobs.(splitobs(tup, at=(.2,.3))) == (3,4,8)
    end
end

# @testset "ObsView" begin
#     A = ObsView(X)
#     b, c = @inferred splitobs(A, .7)
#     @test b isa ObsView
#     @test c isa ObsView
#     @test b == A[1:105]
#     @test c == A[106:end]
# end

# @testset "BatchView" begin
#     A = BatchView(X, 5)
#     b, c = @inferred splitobs(A, .6)
#     @test b isa BatchView
#     @test c isa BatchView
#     @test b == A[1:18]
#     @test c == A[19:end]
# end
