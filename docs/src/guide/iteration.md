```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Iteration & Data Loaders

Once a data container is shuffled, split and transformed, the final step of most
pipelines is to *iterate* over it — usually in mini-batches, often shuffled
anew each epoch, and sometimes loaded in parallel. This page covers the
iteration tools, from the high-level [`DataLoader`](@ref) down to the batching
primitives it is built on.

## `eachobs` and `DataLoader`

[`eachobs`](@ref)`(data)` returns an iterator over the observations of `data`.
By default it yields one observation at a time:

```julia
X = rand(4, 100)

for x in eachobs(X)
    # entered 100 times, each x is a length-4 vector
end
```

Pass `batchsize` to iterate over mini-batches instead. The last dimension of
each array is the one split into batches:

```julia
for x in eachobs(X; batchsize=10)
    # entered 10 times, each x is a 4×10 matrix
end
```

`eachobs` is a thin wrapper around [`DataLoader`](@ref), which is the object you
will reach for in training loops. A `DataLoader` supports shuffling, dropping
the last partial batch, parallel loading and buffered loading, all through
keyword arguments:

```julia
Xtrain = rand(10, 100)
Ytrain = rand('a':'z', 100)

loader = DataLoader((data=Xtrain, label=Ytrain); batchsize=5, shuffle=true)

for epoch in 1:100
    for (x, y) in loader        # destructure the named tuple
        @assert size(x) == (10, 5)
        @assert size(y) == (5,)
        # update the model on this mini-batch
    end
end
```

The most useful keyword arguments are:

- **`batchsize`**: observations per mini-batch. If negative, iterate over single
  observations. Default `1`.
- **`shuffle`**: reshuffle the observations *at the start of every epoch*. This
  differs from wrapping the data in [`shuffleobs`](@ref), which fixes a single
  permutation for the lifetime of the view. Default `false`.
- **`partial`**: when the number of observations is not divisible by
  `batchsize`, keep (`true`) or drop (`false`) the smaller last batch. Default
  `true`.
- **`parallel`**: load batches on worker threads. Requires starting Julia with
  multiple threads (`Threads.nthreads() > 1`). Speeds up loading when `getobs`
  is expensive (e.g. reading from disk), but **breaks ordering guarantees**.
  Default `false`.
- **`buffer`**: reuse a preallocated buffer across iterations via
  [`getobs!`](@ref), avoiding per-batch allocations. Default `false`.
- **`collate`**: how a vector of observations is combined into a batch. See the
  next section.

See the [`DataLoader`](@ref) docstring for the complete list.

## `BatchView`: batches as an indexable vector

While `DataLoader` is an iterator, [`BatchView`](@ref) presents the same batched
view of the data as an *indexable* vector of batches. This is handy when you
need random access to batches, or to know `length` up front:

```jldoctest
julia> bv = BatchView(collect(1:10); batchsize=3);

julia> length(bv)
4

julia> bv[1]
3-element Vector{Int64}:
 1
 2
 3
```

With the default `partial=true` the last batch is smaller (here it holds the
single observation `10`); set `partial=false` to drop it instead.

### Collation

By default a batch is formed by `getobs(data, indices)`. The `collate` keyword
(shared by `BatchView` and `DataLoader`) lets you change this:

- `collate=nothing` (default): the batch is `getobs(data, indices)`.
- `collate=false`: the batch is the *vector* `[getobs(data, i) for i in indices]`,
  leaving the observations uncombined.
- `collate=true`: apply [`batch`](@ref) to that vector, recursively stacking
  arrays along a new last dimension.
- `collate=f`: use your own function `f` on the vector of observations.

A custom collate function is the idiomatic way to handle observations of varying
size (for example, padding variable-length sequences into a single padded
batch).

## Single random observations

To draw observations uniformly at random — for example to peek at the data or to
implement a custom sampler — use [`randobs`](@ref):

```julia
randobs(X)        # one random observation
randobs(X, 5)     # a batch of 5 random observations
```

## Sliding windows

For sequential data, [`slidingwindow`](@ref) provides a vector-like view whose
elements are fixed-size windows of adjacent observations. The `stride`
determines the gap between the start of consecutive windows:

```jldoctest
julia> s = slidingwindow(1:10; size=3, stride=2);

julia> s[1]
1:3

julia> s[2]
3:5

julia> [collect(w) for w in s]
4-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [3, 4, 5]
 [5, 6, 7]
 [7, 8, 9]
```

Only complete windows are included, so trailing observations that do not fill a
window are dropped. As with everything else, windows are not materialized until
indexed or passed to [`getobs`](@ref).

## Batching primitives

Under the iteration machinery sit a handful of plain functions for assembling
and disassembling batches, which are useful on their own.

[`batch`](@ref) stacks a vector of observations into a single array along a new
trailing dimension, and [`unbatch`](@ref) is its inverse:

```jldoctest
julia> batch([[1, 2], [3, 4], [5, 6]])
2×3 Matrix{Int64}:
 1  3  5
 2  4  6

julia> unbatch([1 3 5; 2 4 6])
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
```

[`chunk`](@ref) splits a collection into a number of contiguous chunks, either
by chunk `size` or by the number of chunks `n`:

```jldoctest
julia> chunk(1:10; size=3)
4-element Vector{UnitRange{Int64}}:
 1:3
 4:6
 7:9
 10:10
```

Related helpers include [`batchseq`](@ref) and [`batch_sequence`](@ref) for
padding and batching variable-length sequences. See the
[API Reference](@ref) for the full list.

## Where to go next

- [Data Containers](@ref) — the interface all of this is built on.
- [API Reference](@ref) — the complete list of exported functions.
