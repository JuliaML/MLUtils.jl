```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Data Subsets and Views

Machine learning pipelines spend a lot of time *rearranging* observations:
shuffling them, splitting them into training and test sets, selecting a subset
for a fold. Doing this by copying data around is wasteful, especially for large
datasets. MLUtils.jl instead represents these operations **lazily**, as views
that only remember *which* observations they refer to. No data is copied until
you explicitly ask for it with [`getobs`](@ref).

## `obsview`: the building block

[`obsview`](@ref)`(data, indices)` creates a view of the observations of `data`
at the given `indices`. It is the primitive on top of which shuffling,
splitting and folding are built.

For arrays, `obsview` returns a `SubArray` (a standard Julia view) along the
observation dimension, so no memory is allocated:

```jldoctest
julia> data = [1 2 3; 4 5 6]
2×3 Matrix{Int64}:
 1  2  3
 4  5  6

julia> obsview(data, [1, 3])
2×2 view(::Matrix{Int64}, :, [1, 3]) with eltype Int64:
 1  3
 4  6
```

For data containers that have no natural lazy representation, `obsview` returns
an [`ObsView`](@ref) wrapper, which itself behaves like a data container (it
supports `numobs` and `getobs`) and can be passed to any other function in the
package. You can customize the behavior of `obsview` on your own types by
implementing a method for it.

### Choosing the observation dimension

By default the last dimension of an array is treated as the observation
dimension. Sometimes that is not what you want. A typical example is the
3D arrays fed to recurrent networks and transformers, which usually have shape
`(n_features, n_timesteps, n_samples)`. If you want to treat the *timesteps* as
observations, pass an [`ObsDim`](@ref) object:

```jldoctest
julia> data = reshape([1:24;], 3, 4, 2)
3×4×2 Array{Int64, 3}:
[:, :, 1] =
 1  4  7  10
 2  5  8  11
 3  6  9  12

[:, :, 2] =
 13  16  19  22
 14  17  20  23
 15  18  21  24

julia> ov = obsview(data, ObsDim(2));

julia> numobs(ov)
4

julia> getobs(ov, 1)
3×2 Matrix{Int64}:
 1  13
 2  14
 3  15
```

## Shuffling

[`shuffleobs`](@ref)`(data)` returns the dataset reordered by a random
permutation. Like `obsview`, it only shuffles indices — the underlying values
are not copied:

```jldoctest
julia> typeof(shuffleobs(rand(4, 10))) <: SubArray
true
```

Because datasets are often stored sorted by label, shuffling before splitting
is almost always what you want. For reproducibility you can pass a random
number generator as the first argument:

```julia
using Random

shuffleobs(MersenneTwister(42), data)
```

When `data` is a tuple of arrays, the *same* permutation is applied to each
element, keeping features and labels aligned:

```julia
Xs, Ys = shuffleobs((X, Y))   # X and Y are shuffled together
```

## Splitting into train/test/validation

[`splitobs`](@ref) partitions a data container into two or more disjoint
subsets. The `at` argument controls the split:

- a number in `(0, 1)` gives the *proportion* of observations in the first
  subset;
- an integer gives the *number* of observations in the first subset;
- a tuple gives the size of each subset except the last, which receives the
  remainder. The number of returned subsets is `length(at) + 1`.

```jldoctest
julia> train, test = splitobs(collect(1:10); at=0.6);

julia> train
6-element view(::Vector{Int64}, 1:6) with eltype Int64:
 1
 2
 3
 4
 5
 6

julia> test
4-element view(::Vector{Int64}, 7:10) with eltype Int64:
  7
  8
  9
 10
```

A three-way split (e.g. train/validation/test) is just as easy:

```jldoctest
julia> data = (x = ones(2, 10), n = 1:10);

julia> train, val, test = splitobs(data; at=(0.5, 0.3));

julia> map(numobs, (train, val, test))
(5, 3, 2)
```

`splitobs` splits *contiguous* ranges by default, so combine it with
shuffling when the data is ordered. You can either shuffle first or pass
`shuffle=true` directly:

```julia
train, test = splitobs((X, Y); at=0.7, shuffle=true)
```

### Stratified splitting

When classes are imbalanced, a plain split can leave a subset with too few (or
zero) examples of a rare class. Passing `stratified` — a vector of labels with
one entry per observation — makes `splitobs` preserve the per-class proportions
in every subset:

```jldoctest
julia> splitobs(1:10; at=0.5, stratified=[0,0,0,0,1,1,1,1,1,1])
([1, 2, 5, 6, 7], [3, 4, 8, 9, 10])
```

Here the two zeros and three ones of each class are distributed evenly: the
first subset gets 2 zeros and 3 ones, and so does the second.

## A note on laziness

Because all of the above return views, chaining them is cheap. The snippet

```julia
cv_data, test_data = splitobs(shuffleobs((X, Y)); at=0.85)
```

shuffles and splits a labeled dataset of arbitrary size without copying a single
observation. The actual data is only touched when you call [`getobs`](@ref) — or
when you iterate with a [`DataLoader`](@ref), which calls it for you.

## Where to go next

- [Lazy Transformations](@ref) — apply functions, filters and groupings on top
  of these views.
- [Cross-validation](@ref) — repartition a dataset into folds.
