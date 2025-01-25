```@meta
CollapsedDocStrings = true
```

# API Reference


## Core API

```@docs
getobs
getobs!
numobs
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
batchsize
batchseq
BatchView
eachobs
DataLoader
obsview
ObsView
randobs
```

## Partitioning

```@docs
leavepout
kfolds
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
chunk
flatten
group_counts
group_indices
normalise
rpad_constant
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
