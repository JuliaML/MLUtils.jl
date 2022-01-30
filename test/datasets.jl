@testset "load_iris" begin
    xtmp, ytmp, vars = load_iris()

    @test typeof(xtmp) <: Matrix{Float64}
    @test typeof(ytmp) <: Vector{String}
    @test typeof(vars) <: Vector{String}
    @test size(xtmp) == (4, 150)
    @test length(ytmp) == 150
    @test length(vars) == size(xtmp, 1)
    @test mean(xtmp, dims=2) ≈ [5.843333333333333333, 3.05733333333333333, 3.758000, 1.199333333333333]
    @test mean(xtmp[:,1:50], dims=2) ≈ [5.006, 3.428, 1.462, 0.246]
    @test mean(xtmp[:,51:100], dims=2) ≈ [5.936, 2.77, 4.26, 1.326]
end

@testset "load_sin" begin
    xtmp, ytmp, vars = load_sin()

    @test typeof(xtmp) <: Vector{Float64}
    @test typeof(ytmp) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(xtmp) == length(ytmp) == 15
    @test length(vars) == 2
end

@testset "load_line" begin
    xtmp, ytmp, vars = load_line()

    @test typeof(xtmp) <: Vector{Float64}
    @test typeof(ytmp) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(xtmp) == length(ytmp) == 11
    @test length(vars) == 2
end

@testset "load_poly" begin
    xtmp, ytmp, vars = load_poly()

    @test typeof(xtmp) <: Vector{Float64}
    @test typeof(ytmp) <: Vector{Float64}
    @test typeof(vars) <: Vector{String}
    @test length(xtmp) == length(ytmp) == 50
    @test length(vars) == 2
end

@testset "load_spiral" begin
    xtmp, ytmp, vars = load_spiral()
    @test typeof(xtmp) <: Matrix{Float64}
    @test typeof(ytmp) <: Vector{Int}
    @test typeof(vars) <: Vector{String}
    @test length(xtmp[:, 1]) == length(ytmp) == 194
    @test length(vars) == 3
end
