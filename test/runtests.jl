using MLUtils
using SparseArrays
using Random, Statistics
using Test

showcompact(io, x) = show(IOContext(io, :compact => true), x)

@testset "MLUtils.jl" begin
    @testset "observation" begin; include("observation.jl"); end
    @testset "randobs" begin; include("randobs.jl"); end
    @testset "datasubset" begin; include("datasubset.jl"); end
    @testset "splitobs" begin; include("datasubset.jl"); end
    @testset "shuffleobs" begin; include("datasubset.jl"); end
end
