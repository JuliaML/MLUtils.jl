@test_throws DimensionMismatch splitobs((X, rand(149)))

@testset "typestability" begin
    @testset "Int" begin
        @test_throws ArgumentError splitobs(10, 0.)
        @test_throws ArgumentError splitobs(10, 1.)
        @test_throws ArgumentError splitobs(10, (0.2,0.0))
        @test_throws ArgumentError splitobs(10, (0.2,0.8))
        @test typeof(@inferred(splitobs(10))) <: NTuple{2}
        @test eltype(@inferred(splitobs(10))) <: UnitRange
        @test typeof(@inferred(splitobs(10, 0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(10, (0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(10, 0.5))) <: UnitRange
        @test eltype(@inferred(splitobs(10, (0.5,0.2)))) <: UnitRange
        @test eltype(@inferred(splitobs(10, at=0.5))) <: UnitRange
    end
    for var in vars
        @test_throws ArgumentError splitobs(var, 0.)
        @test_throws ArgumentError splitobs(var, 1.)
        @test_throws ArgumentError splitobs(var, (0.2,0.0))
        @test_throws ArgumentError splitobs(var, (0.2,0.8))
        @test typeof(@inferred(splitobs(var))) <: NTuple{2}
        @test eltype(@inferred(splitobs(var))) <: SubArray
        @test typeof(@inferred(splitobs(var, 0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(var, (0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(var, 0.5))) <: SubArray
        @test eltype(@inferred(splitobs(var, (0.5,0.2)))) <: SubArray
    end
    for tup in tuples
        @test_throws ArgumentError splitobs(tup, 0.)
        @test_throws ArgumentError splitobs(tup, 1.)
        @test_throws ArgumentError splitobs(tup, (0.2,0.0))
        @test_throws ArgumentError splitobs(tup, (0.2,0.8))
        @test typeof(@inferred(splitobs(tup, 0.5))) <: NTuple{2}
        @test typeof(@inferred(splitobs(tup, (0.5,0.2)))) <: NTuple{3}
        @test eltype(@inferred(splitobs(tup, 0.5))) <: Tuple
        @test eltype(@inferred(splitobs(tup, (0.5,0.2)))) <: Tuple
    end
end

@testset "Int" begin
    @test splitobs(10) == (1:7,8:10)
    @test splitobs(10, 0.5) == (1:5,6:10)
    @test splitobs(10, (0.5,0.3)) == (1:5,6:8,9:10)
    @test splitobs(150) == splitobs(150, 0.7)
    @test splitobs(150, at=0.5) == splitobs(150, 0.5)
    @test splitobs(150, at=(0.5,0.2)) == splitobs(150, (0.5,0.2))
    @test nobs.(splitobs(150)) == (105,45)
    @test nobs.(splitobs(150, at=(.2,.3))) == (30,45,75)
    @test nobs.(splitobs(150, at=(.1,.2,.3))) == (15,30,45,60)
    # tests if all obs are still present and none duplicated
    @test sum(sum.(getobs.(splitobs(150)))) == 11325
    @test sum(sum.(splitobs(150,at=.1))) == 11325
    @test sum(sum.(splitobs(150,at=(.2,.1)))) == 11325
    @test sum(sum.(splitobs(150,at=(.1,.4,.2)))) == 11325
    @test sum.(splitobs(150)) == (5565, 5760)
end

println("<HEARTBEAT>")

@testset "Array, SparseArray, and SubArray" begin
    for var in (Xs, ys, vars...)
        @test nobs.(splitobs(var, at=0.7)) == (105,45)
        @test nobs.(splitobs(var, at=(.2,.3))) == (30,45,75)
        @test nobs.(splitobs(var, at=(.1,.2,.3))) == (15,30,45,60)
    end
    # tests if all obs are still present and none duplicated
    @test sum(vec.(sum.(getobs.(splitobs(sparse(X1))),dims=2))) == fill(11325,10)
    @test sum(vec.(sum.(splitobs(X1),dims=2))) == fill(11325,10)
    @test sum(vec.(sum.(splitobs(X1,at=.1),dims=2))) == fill(11325,10)
    @test sum(vec.(sum.(splitobs(X1,at=(.2,.1)),dims=2))) == fill(11325,10)
    @test sum(vec.(sum.(splitobs(X1,at=(.1,.4,.2)),dims=2))) == fill(11325,10)
    @test sum(vec.(sum.(getobs.(splitobs(sparse(X1),at=(.2,.1))),dims=2))) == fill(11325,10)
    @test sum.(splitobs(Y1)) == (5565, 5760)
    @test sum.(getobs.(splitobs(sparse(Y1)))) == (5565, 5760)
end

println("<HEARTBEAT>")

@testset "Tuple of Array, SparseArray, and SubArray" begin
    for tup in ((Xs,ys), (X,ys), (Xs,y), (Xs,Xs), (XX,X,ys), (X,yv), (Xv,y), tuples...)
        @test_throws MethodError splitobs(tup..., 0.5)
        @test_throws MethodError splitobs(tup...)
        @test all(map(x->(typeof(x)<:Tuple), splitobs(tup)))
        @test all(map(x->(typeof(x)<:Tuple), splitobs(tup,at=0.5)))
        @test nobs.(splitobs(tup)) == (105,45)
        @test nobs.(splitobs(tup, at=(.2,.3))) == (30,45,75)
        @test nobs.(splitobs(tup, at=(.1,.2,.3))) == (15,30,45,60)
    end
    # tests if all obs are still present and none duplicated
    # also tests that both paramter are split disjoint
    train,test = splitobs((X1,Y1,X1))
    @test vec(sum(train[1],dims=2)+sum(test[1],dims=2)) == fill(11325,10)
    @test vec(sum(train[3],dims=2)+sum(test[3],dims=2)) == fill(11325,10)
    @test sum(train[2]) + sum(test[2]) == 11325
    @test all(train[1]' .== train[2])
    @test all(train[3]' .== train[2])
    @test all(test[1]' .== test[2])
    @test all(test[3]' .== test[2])
    @test vec(sum(train[1],dims=1)) == fill(5565,10)
    @test vec(sum(test[1],dims=1)) == fill(5760,10)
    @test sum(train[2]) == 5565
    @test sum(test[2]) == 5760
    @test all(train[1] .== train[2])
    @test all(test[1] .== test[2])
    train,test = splitobs((sparse(X1),Y1),at=0.2)
    @test vec(sum(getobs(train[1]),dims=2)+sum(getobs(test[1]),dims=2)) == fill(11325,10)
    @test sum(train[2]) + sum(test[2]) == 11325
    @test all(getobs(train[1])' .== train[2])
    @test all(getobs(test[1])' .== test[2])
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
