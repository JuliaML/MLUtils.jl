```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Iteration & Views

Once a data container is shuffled, split and transformed, the final step of most
pipelines is to *iterate* over it — usually in mini-batches. The workhorse for that
is the [`DataLoader`](@ref), which has its own page: [Data Loaders](@ref). This page
covers the lighter-weight iteration and view primitives that surround it —
[`eachobs`](@ref), [`BatchView`](@ref), [`randobs`](@ref), [`slidingwindow`](@ref),
and the plain batching functions underneath.

## `eachobs`

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

`eachobs` is a thin convenience wrapper around [`DataLoader`](@ref) and forwards the
same keyword arguments (`shuffle`, `parallel`, `collate`, …). When you need the full
set of options — especially shuffling anew each epoch or parallel/distributed loading
— construct a [`DataLoader`](@ref) directly; see [Data Loaders](@ref).

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
single observation `10`); set `partial=false` to drop it instead. `BatchView` also
accepts the `collate` keyword, with the same meaning as for [`DataLoader`](@ref)
(see [Collation](@ref)).

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

- [Data Loaders](@ref) — the full `DataLoader`, collation, and parallel/distributed
  loading.
- [Data Containers](@ref) — the interface all of this is built on.
- [API Reference](@ref) — the complete list of exported functions.
