@test_throws DimensionMismatch splitobs((X, rand(149)), at=0.7)

### These tests pass on julia 1.6 but fail on higher versions
# @testset "typestability" begin
#     @testset "Int" begin
#         @test typeof(@inferred(splitobs(10, at=0.))) <: NTuple{2}
#         @test eltype(@inferred(splitobs(10, at=0.))) <: UnitRange
#         @test typeof(@inferred(splitobs(10, at=1.))) <: NTuple{2}
#         @test eltype(@inferred(splitobs(10, at=1.))) <: UnitRange
#         @test typeof(@inferred(splitobs(10, at=0.7))) <: NTuple{2}
#         @test eltype(@inferred(splitobs(10, at=0.7))) <: UnitRange
#         @test typeof(@inferred(splitobs(10, at=0.5))) <: NTuple{2}
#         @test typeof(@inferred(splitobs(10, at=(0.5,0.2)))) <: NTuple{3}
#         @test eltype(@inferred(splitobs(10, at=0.5))) <: UnitRange
#         @test eltype(@inferred(splitobs(10, at=(0.5,0.2)))) <: UnitRange
#         @test eltype(@inferred(splitobs(10, at=0.5))) <: UnitRange
#         @test eltype(@inferred(splitobs(10, at=(0.,0.2)))) <: UnitRange
#     end
#     for var in vars
#         @test typeof(@inferred(splitobs(var, at=0.7))) <: NTuple{2}
#         @test eltype(@inferred(splitobs(var, at=0.7))) <: SubArray
#         @test typeof(@inferred(splitobs(var, at=0.5))) <: NTuple{2}
#         @test typeof(@inferred(splitobs(var, at=(0.5,0.2)))) <: NTuple{3}
#         @test eltype(@inferred(splitobs(var, at=0.5))) <: SubArray
#         @test eltype(@inferred(splitobs(var, at=(0.5,0.2)))) <: SubArray
#     end
#     for tup in tuples
#         @test typeof(@inferred(splitobs(tup, at=0.5))) <: NTuple{2}
#         @test typeof(@inferred(splitobs(tup, at=(0.5,0.2)))) <: NTuple{3}
#         @test eltype(@inferred(splitobs(tup, at=0.5))) <: Tuple
#         @test eltype(@inferred(splitobs(tup, at=(0.5,0.2)))) <: Tuple
#     end
# end

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
    @test sum(sum.(splitobs(X1,at=.1))) == 1200
    @test sum(sum.(splitobs(X1,at=(.2,.1)))) == 1200
    @test sum(sum.(splitobs(X1,at=(.1,.4,.2)))) == 1200
end

@testset "Tuple of Array, SparseArray, and SubArray" begin
    for tup in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys), (X,yv), (Xv,y), tuples...)
        @test_throws MethodError splitobs(tup..., at=0.5)
        @test all(map(x->(typeof(x)<:Tuple), splitobs(tup,at=0.5)))
        @test numobs.(splitobs(tup, at=0.6)) == (9, 6)
        @test numobs.(splitobs(tup, at=(.2,.3))) == (3,4,8)
    end
end

@testset "shuffle" begin
    s = splitobs(X, at=0.1)[1]
    @test s == X[:,1:2]
    s = splitobs(X, at=0.1, shuffle=true)[1]
    @test size(s) == size(X[:,1:2])
    @test s[:,1] !== s[:,2]
    @test s != X[:,1:2]
    @test any(s[:,1] == x for x in eachcol(X))
    @test any(s[:,2] == x for x in eachcol(X))
end
