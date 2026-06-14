```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Data Containers

At the heart of MLUtils.jl lies a simple but powerful abstraction: the
**data container**. A data container is any object `data` that knows

1. how many observations it holds, queried with [`numobs`](@ref)`(data)`, and
2. how to materialize the observation(s) at some index, queried with
   [`getobs`](@ref)`(data, idx)`, where `idx` is an integer in `1:numobs(data)`
   or a vector of such integers.

Almost every function in this package — splitting, shuffling, batching,
resampling, cross-validation — is written against these two functions alone.
As a consequence, the same code works for plain arrays, tuples of arrays,
tables, and your own custom types, with no special casing.

## Built-in data containers

[`numobs`](@ref) and [`getobs`](@ref) are implemented out of the box for the
most common Julia types: arrays, tuples, named tuples, dictionaries, and
[Tables.jl](https://github.com/JuliaData/Tables.jl) tables (such as
`DataFrame`s). You can therefore use these types as data containers directly.

For arrays the convention is that the **last dimension is the observation
dimension** (sometimes also called the batch dimension). Let's start with a
simple matrix, which we interpret as two observations of three features each:

```jldoctest
julia> data = [1 2; 3 4; 5 6]
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> numobs(data)
2

julia> getobs(data, 1)
3-element Vector{Int64}:
 1
 3
 5
```

### Tuples and named tuples

In supervised learning the features and the targets are usually kept in
separate arrays. Grouping them in a tuple (or a named tuple) yields a data
container whose observations are tuples of the corresponding observations.
The number of observations must agree across all elements:

```jldoctest
julia> data = (x = ones(2, 3), y = [1, 2, 3]);

julia> numobs(data)
3

julia> getobs(data, 2)
(x = [1.0, 1.0], y = 2)
```

This is the idiomatic way to keep features and labels in sync: any operation
that reshuffles or subsets the observations will apply the *same* permutation
to every element of the tuple.

### Tables

Tables.jl-compatible tables are data containers where each row is an
observation:

```jldoctest
julia> using DataFrames

julia> data = DataFrame(x = 1:4, y = ["a", "b", "c", "d"])
4×2 DataFrame
 Row │ x      y
     │ Int64  String
─────┼───────────────
   1 │     1  a
   2 │     2  b
   3 │     3  c
   4 │     4  d

julia> numobs(data)
4

julia> getobs(data, 3)
(x = 3, y = "c")
```

## Custom data containers

If you have your own dataset type, you have two ways to make it work with
MLUtils.jl.

The quickest one is to implement `Base.length` and `Base.getindex`: when
`numobs` and `getobs` are not specialized for a type, they fall back to these
two functions. This is convenient because many types already implement them.

```jldoctest customdata
julia> struct DummyDataset
           length::Int
       end

julia> Base.length(d::DummyDataset) = d.length

julia> function Base.getindex(d::DummyDataset, i::Int)
           1 <= i <= d.length || throw(ArgumentError("Index out of bounds"))
           return 10 * i
       end

julia> data = DummyDataset(10);

julia> numobs(data)
10

julia> getobs(data, 2)
20
```

That single pair of methods is enough to use your type with [`DataLoader`](@ref),
[`splitobs`](@ref), [`kfolds`](@ref), and the rest of the package.

Alternatively, you can directly extend [`numobs`](@ref) and [`getobs`](@ref).
This is the preferred route when `length`/`getindex` already carry a different
meaning for your type, or when loading an observation requires logic that is
distinct from ordinary indexing (for instance, reading an image from disk):

```julia
using MLUtils
import MLUtils: numobs, getobs

struct ImageFolder
    paths::Vector{String}
end

numobs(d::ImageFolder) = length(d.paths)
getobs(d::ImageFolder, i::Int) = load_image(d.paths[i])
```

### In-place loading with `getobs!`

For performance-critical pipelines it can be worthwhile to load an observation
into a preallocated buffer instead of allocating a fresh array every time. This
is what [`getobs!`](@ref) is for. When you implement `getobs!(buffer, data, idx)`
for your type, [`DataLoader`](@ref) can reuse a buffer across iterations by
passing `buffer=true` (see [Iteration & Data Loaders](@ref)).

If you do not implement it, `getobs!` falls back to `getobs`, so it is purely
an optional optimization.

## Where to go next

- [Data Subsets and Views](@ref) — lazily subset, shuffle and split data
  without copying.
- [Lazy Transformations](@ref) — map, filter, group and join data containers.
- [Iteration & Data Loaders](@ref) — iterate over observations and mini-batches.
