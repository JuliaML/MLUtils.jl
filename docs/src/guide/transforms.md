```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Lazy Transformations

On top of plain subsetting, MLUtils.jl offers a small algebra of **lazy
transformations** that take a data container and return a new data container.
None of them touch the underlying data at construction time; the transformation
is applied on demand, when you index the result or call [`getobs`](@ref) on it.
This makes them cheap to compose and ideal for expressing data preprocessing
and augmentation pipelines.

## `mapobs`: apply a function to each observation

[`mapobs`](@ref)`(f, data)` lazily maps `f` over the observations of `data`:

```jldoctest
julia> data = [1, 2, 3, 4];

julia> mdata = mapobs(x -> x .^ 2, data);

julia> getobs(mdata, 1:4)
4-element Vector{Int64}:
  1
  4
  9
 16
```

Here `f` is applied to a whole batch of observations at once (the default
`batched=:auto`), which is why we broadcast with `.^`. See
[Controlling batched application](@ref) below for the alternatives.

`f` receives whatever `getobs(data, idx)` returns, so it composes naturally with
tuples and named tuples. Here we derive two new fields from an observation:

```jldoctest
julia> data = (a = [1, 2, 3], b = [1, 2, 3]);

julia> mdata = mapobs(data) do x
           (c = x.a .+ x.b, d = x.a .- x.b)
       end;

julia> mdata[1]
(c = 2, d = 0)

julia> mdata[1:2]
(c = [2, 4], d = [0, 0])
```

This is the natural place to put **data augmentation**: wrap the training data
in a `mapobs` that perturbs each observation, and the perturbation is recomputed
every time the observation is accessed.

### A named tuple of functions

Passing a `NamedTuple` of functions builds a data container of named tuples,
and lets you address each "column" by field name:

```jldoctest
julia> nameddata = mapobs((x = sqrt, y = log), 1:10);

julia> getobs(nameddata, 4)
(x = 2.0, y = 1.3862943611198906)

julia> getobs(nameddata.x, 4)
2.0
```

### Controlling batched application

By default (`batched=:auto`), `mapobs` calls `f` both on single observations and
on batches, leaving it up to `f` to handle each case. If your function is
written for one specific case, you can force the behavior with `batched=:never`
(always call `f` per observation) or `batched=:always` (always call `f` on a
batch). See the [`mapobs`](@ref) docstring for the precise semantics.

## `filterobs`: keep a subset of observations

[`filterobs`](@ref)`(f, data)` returns a view containing only the observations
for which `f(getobs(data, i))` is `true`:

```jldoctest
julia> data = 1:10;

julia> fdata = filterobs(>(5), data);

julia> numobs(fdata)
5

julia> getobs(fdata, 1:numobs(fdata))
5-element Vector{Int64}:
  6
  7
  8
  9
 10
```

## `groupobs`: split observations into groups

[`groupobs`](@ref)`(f, data)` partitions a data container into several data
containers, grouping observations by the value of `f(obs)`. It returns a
dictionary keyed by group:

```jldoctest
julia> data = 1:6;

julia> groups = groupobs(iseven, data);

julia> sort(collect(keys(groups)))
2-element Vector{Bool}:
 0
 1

julia> getobs(groups[true], 1:numobs(groups[true]))
3-element Vector{Int64}:
 2
 4
 6
```

## `joinobs`: concatenate data containers

[`joinobs`](@ref)`(datas...)` lazily concatenates several data containers into
one, as if their observations had been stacked end to end:

```jldoctest
julia> jdata = joinobs([1, 2, 3], [4, 5]);

julia> numobs(jdata)
5

julia> getobs(jdata, 1:5)
5-element Vector{Int64}:
 1
 2
 3
 4
 5
```

## Composing a pipeline

Because every transformation returns a data container, you can stack them and
mix them with the subsetting tools from the previous page. For instance, to
build an augmented training stream from a labeled, shuffled, split dataset:

```julia
train, test = splitobs(shuffleobs((X, Y)); at=0.8)

augmented = mapobs(train) do (x, y)
    (x .+ 0.1f0 .* randn(Float32, size(x)), y)
end

for (xb, yb) in eachobs(augmented; batchsize=32)
    # train on the freshly augmented mini-batch
end
```

Nothing in this pipeline materializes data until the inner `eachobs` loop runs
[`getobs`](@ref) on each mini-batch.

## Where to go next

- [Labeled Data and Resampling](@ref) — balance class distributions.
- [Iteration & Data Loaders](@ref) — turn these containers into mini-batch
  streams.
