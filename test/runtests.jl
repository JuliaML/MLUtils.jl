using MLBase
using SparseArrays
using Test

@testset "MLBase.jl" begin
    @testset "observation" begin
        include("observation.jl")
    end
end
