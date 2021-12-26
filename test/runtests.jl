using MLUtils
using SparseArrays
using Test

@testset "MLUtils.jl" begin
    @testset "observation" begin
        include("observation.jl")
    end
end
