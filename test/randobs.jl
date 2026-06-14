x = randobs(X[:,1:4])
@test size(x) == (4,)
@test any(X[:,i] == x for i in 1:4) 

x = randobs(X[:,1:4], 2)
@test size(x) == (4, 2)
@test any(X[:,i] == x[:,1] for i in 1:4)
@test any(X[:,i] == x[:,2] for i in 1:4)

@testset "rng" begin
    x = randobs(MersenneTwister(0), X)
    @test size(x) == (4,)
    @test any(X[:,i] == x for i in 1:15)
    @test randobs(MersenneTwister(0), X) == randobs(MersenneTwister(0), X)

    x = randobs(MersenneTwister(0), X, 3)
    @test size(x) == (4, 3)
    @test randobs(MersenneTwister(0), X, 3) == randobs(MersenneTwister(0), X, 3)
end
