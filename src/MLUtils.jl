module MLUtils

using Random
using Statistics
using ShowCases: ShowLimit
using FLoops: @floop
using FLoops.Transducers: Executor, ThreadedEx
import StatsBase: sample
using Transducers
using Tables
using DataAPI
using Base: @propagate_inbounds
using Random: AbstractRNG, shuffle!, rand!, randn!
import ChainRulesCore: rrule
using ChainRulesCore: @non_differentiable, unthunk, AbstractZero,
                      NoTangent, ZeroTangent, ProjectTo

using SimpleTraits
import NNlib

@traitdef IsTable{X}
@traitimpl IsTable{X} <- Tables.istable(X)
    
using Compat: stack

include("observation.jl")
export numobs,
       getobs,
       getobs!

include("obstransform.jl")
export mapobs,
       filterobs,
       groupobs,
       joinobs,
       shuffleobs

include("batchview.jl")
export batchsize, BatchView

include("obsview.jl")
export obsview, ObsView

include("dataloader.jl")
export eachobs, DataLoader

include("parallel.jl")

include("folds.jl")
export kfolds,
       leavepout

include("randobs.jl")
export randobs

include("resample.jl")
export oversample,
       undersample

include("slidingwindow.jl")
export slidingwindow

include("splitobs.jl")
export splitobs

include("batch.jl")
export batch,
       batchseq,
       batch_sequence,
       unbatch

include("utils.jl")
export chunk,
       falses_like,
       fill_like,
       flatten,
       group_counts,
       group_indices,
       normalise,
       ones_like,
       rand_like,
       randn_like,
       rpad_constant,
       stack, # in Base since julia v1.9
       trues_like,
       unsqueeze,
       unstack,
       zeros_like

include("Datasets/Datasets.jl")
using .Datasets
export Datasets,
       load_iris

include("deprecations.jl")

end
