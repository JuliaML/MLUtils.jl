using MLUtils
using MLUtils.Datasets
using MLUtils: RingBuffer, eachobsparallel
using SparseArrays
using Random, Statistics
using Test
using FoldsThreads: TaskPoolEx
using ChainRulesTestUtils: test_rrule
using Zygote: ZygoteRuleConfig
using ChainRulesCore: rrule_via_ad

showcompact(io, x) = show(IOContext(io, :compact => true), x)


# --------------------------------------------------------------------
# create some test data
Random.seed!(1335)
const X = rand(4, 15)
const y = repeat(["setosa","versicolor","virginica"], inner = 5)
const Y = permutedims(hcat(y,y), [2,1])
const Yt = hcat(y,y)
const yt = Y[1:1,:]
const Xv = view(X,:,:)
const yv = view(y,:)
const XX = rand(20,30,15)
const XXX = rand(3,20,30,15)
const vars = (X, Xv, yv, XX, XXX, y)
const tuples = ((X,y), (X,Y), (XX,X,y), (XXX,XX,X,y))
const Xs = sprand(10, 15, 0.5)
const ys = sprand(15, 0.5)
const X1 = hcat((1:15 for i = 1:10)...)'
const Y1 = collect(1:15)

struct EmptyType end

struct FallbackType end
Base.getindex(::FallbackType, i) = 1234
Base.length(::FallbackType) = 5678

struct CustomType end
MLUtils.numobs(::CustomType) = 15
MLUtils.getobs(::CustomType, i::Int) = i
MLUtils.getobs(::CustomType, i::AbstractVector) = collect(i)

struct CustomArray{T,N} <: AbstractArray{T,N}
    length::Int
    size::NTuple{N, Int}

    function CustomArray{T}(n...) where T
        N = length(n)
        new{T,N}(prod(n), Tuple(n))
    end
end
CustomArray(n...) = CustomArray{Float32}(n...)
Base.length(x::CustomArray) = x.length
Base.size(x::CustomArray) = x.size
Base.getindex(x::CustomArray{T}, i::Int) where T = T(1)
Base.getindex(x::CustomArray{T, 1}, i::AbstractVector{Int}) where T = CustomArray{T}(length(i))

struct CustomArrayNoView{T,N} <: AbstractArray{T,N}
    length::Int
    size::NTuple{N, Int}

    function CustomArrayNoView{T}(n...) where T
        N = length(n)
        new{T,N}(prod(n), Tuple(n))
    end
end
CustomArrayNoView(n...) = CustomArrayNoView{Float32}(n...)
Base.length(x::CustomArrayNoView) = x.length
Base.size(x::CustomArrayNoView) = x.size
Base.getindex(x::CustomArrayNoView{T}, i::Int) where T = T(1)
Base.getindex(x::CustomArrayNoView{T, 1}, i::AbstractVector{Int}) where T = CustomArrayNoView{T}(length(i))
Base.view(x::CustomArrayNoView, i...) = error("view not allowed")

struct CustomRangeIndex
    n::Int
end
Base.length(r::CustomRangeIndex) = r.n
Base.getindex(r::CustomRangeIndex, idx::Int) = idx
Base.getindex(r::CustomRangeIndex, idxs::UnitRange) = idxs

# --------------------------------------------------------------------

include("test_utils.jl")

# @testset "MLUtils.jl" begin

@testset "batchview" begin; include("batchview.jl"); end
@testset "eachobs" begin; include("eachobs.jl"); end
@testset "dataloader" begin; include("dataloader.jl"); end
@testset "folds" begin; include("folds.jl"); end
@testset "observation" begin; include("observation.jl"); end
@testset "obsview" begin; include("obsview.jl"); end
@testset "obstransform" begin; include("obstransform.jl"); end
@testset "randobs" begin; include("randobs.jl"); end
@testset "resample" begin; include("resample.jl"); end
@testset "splitobs" begin; include("splitobs.jl"); end
@testset "utils" begin; include("utils.jl"); end
@testset "eachobsparallel" begin; include("parallel.jl"); end

@testset "Datasets/datasets" begin; include("Datasets/datasets.jl"); end
@testset "Datasets/generators" begin; include("Datasets/generators.jl"); end

# end
