```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Cross-validation

To estimate how well a model generalizes, it is common to repeatedly partition
the data into a part used for training and a part held out for validation. The
two classic strategies — *k-fold* and *leave-p-out* — are provided by
[`kfolds`](@ref) and [`leavepout`](@ref). Both return **lazy** iterators over
`(train, validation)` pairs of data subsets, so no data is copied until you call
[`getobs`](@ref).

## k-fold cross-validation

[`kfolds`](@ref)`(data, k)` divides `data` into `k` roughly equal parts. Each
part serves as the validation set exactly once, while the other `k-1` parts form
the training set, producing `k` different partitions:

```julia
for (x_train, x_val) in kfolds(X; k=5)
    model = train(x_train)
    evaluate(model, x_val)
end
```

Every observation lands in the validation set exactly once across the `k`
iterations, so the union of all validation sets reproduces the full dataset.
When the number of observations is not divisible by `k`, the remainder is spread
across the first few folds, so validation sizes may differ by one.

It is often instructive to look at the raw index assignment, which the
integer method `kfolds(n, k)` returns directly:

```jldoctest
julia> train_idx, val_idx = kfolds(10, 5);

julia> val_idx
5-element Vector{UnitRange{Int64}}:
 1:2
 3:4
 5:6
 7:8
 9:10

julia> train_idx[1]
8-element Vector{Int64}:
  3
  4
  5
  6
  7
  8
  9
 10
```

### Labeled data

Like every other function in the package, `kfolds` works on tuples, so features
and labels are partitioned together:

```julia
for ((x_train, y_train), (x_val, y_val)) in kfolds((X, Y); k=10)
    # ...
end
```

### Shuffling before folding

`kfolds` assigns *contiguous* blocks of observations to each fold. Since
datasets are frequently stored sorted by label, you will usually want to shuffle
first, otherwise a fold might contain only a single class. Wrap the data with
[`shuffleobs`](@ref):

```julia
for (x_train, x_val) in kfolds(shuffleobs(X); k=10)
    # ...
end
```

## Leave-p-out cross-validation

[`leavepout`](@ref)`(data, p)` chooses `k ≈ numobs(data) / p` folds so that each
validation set contains about `p` observations. With the default `p = 1` this is
the well-known **leave-one-out** cross-validation:

```jldoctest
julia> train_idx, val_idx = leavepout(10, 2);

julia> val_idx
5-element Vector{UnitRange{Int64}}:
 1:2
 3:4
 5:6
 7:8
 9:10
```

The data method returns the same kind of lazy iterator as `kfolds`:

```julia
for (train, val) in leavepout(X; p=2)
    # numobs(val) is 2 on each iteration when numobs(X) is divisible by 2,
    # otherwise the first few iterations may have 3.
end
```

## Combining with the rest of the pipeline

A complete model-selection loop typically holds out a test set first, then runs
cross-validation on the remainder:

```julia
# shuffle once, set aside 15% for final testing
cv_data, test_data = splitobs(shuffleobs((X, Y)); at=0.85)

for (train_data, val_data) in kfolds(cv_data; k=10)
    for epoch in 1:nepochs
        for (x, y) in eachobs(train_data; batchsize=32)
            # train
        end
    end
    # validate on val_data
end
# final evaluation on test_data
```

Only the inner `eachobs` loop materializes data; the folds, the split and the
shuffle are all lazy views.

## Where to go next

- [Iteration & Data Loaders](@ref) — iterate over the folds in mini-batches.
- [Data Subsets and Views](@ref) — the `splitobs` / `shuffleobs` primitives used
  above.
