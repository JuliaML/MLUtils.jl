```@meta
CollapsedDocStrings = true
```

# API Reference


## Core API

These functions are defined in [MLCore.jl](https://github.com/JuliaML/MLCore.jl).

```@docs
MLCore.getobs
MLCore.getobs!
MLCore.numobs
```

## Lazy Transforms

```@docs
filterobs
groupobs
joinobs
mapobs
shuffleobs
```

## Batching, Iteration, and Views

```@docs
batch
batch_sequence
batchsize
batchseq
BatchView
eachobs
DataLoader
obsview
ObsDim
ObsView
randobs
slidingwindow
```

## Partitioning

```@docs
leavepout
kfolds
timeseries_kfolds
splitobs
```

## Array Constructors

```@docs
falses_like
fill_like
ones_like
rand_like
randn_like
trues_like
zeros_like
```

## Resampling

```@docs
oversample
undersample
```

## Operations

```@docs
batched_searchsortedfirst
batched_searchsortedlast
chunk
flatten
group_counts
group_indices
normalise
rescale
rpad_constant
topk
unbatch
unsqueeze
unstack
```

## Datasets

```@docs
Datasets.load_iris
Datasets.make_sin
Datasets.make_spiral
Datasets.make_poly
Datasets.make_moons
```
