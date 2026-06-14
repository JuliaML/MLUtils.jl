```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Labeled Data and Resampling

In supervised learning each observation comes with a *label* (or *target*).
A recurring problem is **class imbalance**: when one class dominates, a model
can reach a deceptively low loss by simply ignoring the rare classes. MLUtils.jl
provides resampling tools to rebalance a labeled dataset, plus a few helpers to
inspect how labels are distributed.

Throughout this page a "labeled data container" is just a feature container
together with a vector of labels, one per observation. The labels can be of any
type — integers, strings, symbols.

## Inspecting label distributions

Before resampling it is useful to know how many observations fall into each
class. [`group_counts`](@ref) returns a dictionary mapping each distinct value
to its number of occurrences:

```jldoctest
julia> labels = ["a", "b", "b", "c", "b"];

julia> counts = group_counts(labels);

julia> counts["b"]
3

julia> counts["a"]
1
```

If you need the *positions* of each class rather than just the counts, use
[`group_indices`](@ref), which maps each value to the vector of indices at which
it occurs:

```jldoctest
julia> idxs = group_indices([0, 1, 0, 1, 1]);

julia> idxs[0]
2-element Vector{Int64}:
 1
 3

julia> idxs[1]
3-element Vector{Int64}:
 2
 4
 5
```

These indices are exactly what the resampling routines below use internally.

## Oversampling

[`oversample`](@ref)`(data, classes)` rebalances a dataset by *repeating*
observations from the minority classes until every class is (approximately) as
large as the majority class. The result is a larger dataset. Because it works
through [`obsview`](@ref), the repeated observations are views, not copies.

```jldoctest
julia> X = collect(1:6);

julia> Y = ["a", "b", "b", "b", "b", "a"];

julia> Xo, Yo = oversample(X, Y; shuffle=false);

julia> getobs(Yo, 1:numobs(Yo))
8-element Vector{String}:
 "a"
 "b"
 "b"
 "b"
 "b"
 "a"
 "a"
 "a"
```

The minority class `"a"` (originally 2 observations) has been padded up to 4, to
match `"b"`. By default `oversample` shuffles the result so that the repeated
observations are not all clustered at the end; pass `shuffle=false` (as above)
to keep them in a deterministic order. The `fraction` keyword controls *how*
balanced the result is: `fraction=1` (the default) targets a perfectly balanced
dataset, while `fraction=0.5` only guarantees each class reaches half the size
of the largest class.

## Undersampling

[`undersample`](@ref)`(data, classes)` takes the opposite approach: it *discards*
observations from the majority classes until every class matches the smallest
class. The result is a smaller dataset with no repeats.

```jldoctest
julia> X = collect(1:6);

julia> Y = ["a", "b", "b", "b", "b", "a"];

julia> Xu, Yu = undersample(X, Y; shuffle=false);

julia> getobs(Yu, 1:numobs(Yu))
4-element Vector{String}:
 "a"
 "b"
 "b"
 "a"

julia> getobs(Xu, 1:numobs(Xu))
4-element Vector{Int64}:
 1
 2
 3
 6
```

Both classes now have 2 observations each. With `shuffle=false` the kept
observations stay in their original order.

## Working with tuples

Both functions accept a single tuple instead of separate `data` and `classes`
arguments. In that case the **last element of the tuple is assumed to hold the
labels**, which is convenient when features and targets already travel together:

```julia
X = rand(3, 6)
Y = ["a", "b", "b", "b", "b", "a"]

# equivalent to oversample(X, Y), but returns a single tuple
Xo, Yo = oversample((X, Y))
```

This also works for tables. If `data` is a `DataFrame`, you can pass the label
column directly:

```julia
using DataFrames

df = DataFrame(X1=rand(6), X2=rand(6), Y=[:a, :b, :b, :b, :b, :a])
balanced = getobs(oversample(df, df.Y))   # a balanced DataFrame
```

## Reproducibility

Because resampling is randomized (both in *which* minority observations get
repeated and in the final shuffle), you can pass a random number generator as
the first argument to make the result reproducible:

```julia
using Random

oversample(MersenneTwister(0), X, Y)
undersample(MersenneTwister(0), X, Y)
```

## Stratified splitting

Resampling changes the size of the dataset. If instead you only want to *split*
an imbalanced dataset while preserving its class proportions in each part, use
the `stratified` keyword of [`splitobs`](@ref), described in
[Data Subsets and Views](@ref).

## Where to go next

- [Cross-validation](@ref) — repartition a dataset into folds for model
  selection.
- [Data Subsets and Views](@ref) — stratified train/test splitting.
