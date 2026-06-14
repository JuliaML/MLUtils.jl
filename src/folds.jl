
"""
    kfolds(n::Integer, k = 5) -> Tuple

Compute the train/validation assignments for `k` repartitions of
`n` observations, and return them in the form of two vectors. The
first vector contains the index-vectors for the training subsets,
and the second vector the index-vectors for the validation subsets
respectively. A general rule of thumb is to use either `k = 5` or
`k = 10`. 

Each observation is assigned to the validation subset once (and
only once). Thus, a union over all validation index-vectors
reproduces the full range `1:n`. Note that there is no random
assignment of observations to subsets, which means that adjacent
observations are likely to be part of the same validation subset.

# Examples

```jldoctest
julia> train_idx, val_idx = kfolds(10, 5);

julia> train_idx
5-element Vector{Vector{Int64}}:
 [3, 4, 5, 6, 7, 8, 9, 10]
 [1, 2, 5, 6, 7, 8, 9, 10]
 [1, 2, 3, 4, 7, 8, 9, 10]
 [1, 2, 3, 4, 5, 6, 9, 10]
 [1, 2, 3, 4, 5, 6, 7, 8]

julia> val_idx
5-element Vector{UnitRange{Int64}}:
 1:2
 3:4
 5:6
 7:8
 9:10
```
"""
function kfolds(n::Integer, k::Integer=5)
    2 <= k <= n || throw(ArgumentError("n must be positive and k must to be within 2:$(max(2,n))"))
    # Compute the size of each fold. This is important because
    # in general the number of total observations might not be
    # divisible by k. In such cases it is custom that the remaining
    # observations are divided among the folds. Thus some folds
    # have one more observation than others.
    sizes = fill(floor(Int, n / k), k)
    for i = 1:(n%k)
        sizes[i] = sizes[i] + 1
    end
    # Compute start offset for each fold
    offsets = cumsum(sizes) .- sizes .+ 1
    # Compute the validation indices using the offsets and sizes
    val_indices = map((o, s) -> (o:o+s-1), offsets, sizes)
    # The train indices are then the indicies not in validation
    train_indices = map((o, s) -> vcat(1:o-1, o+s:n), offsets, sizes)
    # We return a tuple of arrays
    return train_indices, val_indices
end

"""
    kfolds(data, k = 5)

Repartition a `data` container `k` times using a `k` folds
strategy and return the sequence of folds as a lazy iterator. 
Only data subsets are created, which means that no actual data is copied until
[`getobs`](@ref) is invoked.

Conceptually, a k-folds repartitioning strategy divides the given
`data` into `k` roughly equal-sized parts. Each part will serve
as validation set once, while the remaining parts are used for
training. This results in `k` different partitions of `data`.

In the case that the size of the dataset is not dividable by the
specified `k`, the remaining observations will be evenly
distributed among the parts.

```julia
for (x_train, x_val) in kfolds(X, k=10)
    # code called 10 times
    # numobs(x_val) may differ up to ±1 over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for ((x_train, y_train), val) in kfolds((X, Y), k=10)
    # ...
end
```

By default the folds are created using static splits. Use
[`shuffleobs`](@ref) to randomly assign observations to the
folds.

```julia
for (x_train, x_val) in kfolds(shuffleobs(X), k=10)
    # ...
end
```

See [`leavepout`](@ref) for a related function.
"""
function kfolds(data, k::Integer)
    n = numobs(data)
    train_indices, val_indices = kfolds(n, k)

    ((obsview(data, itrain), obsview(data, ival))
     for (itrain, ival) in zip(train_indices, val_indices))
end

kfolds(data; k) = kfolds(data, k)

"""
    leavepout(n::Integer, [size = 1]) -> Tuple

Compute the train/validation assignments for `k ≈ n/size`
repartitions of `n` observations, and return them in the form of
two vectors. The first vector contains the index-vectors for the
training subsets, and the second vector the index-vectors for the
validation subsets respectively. Each validation subset will have
either `size` or `size+1` observations assigned to it. The
following code snippet generates the index-vectors for `size = 2`.

```julia-repl
julia> train_idx, val_idx = leavepout(10, 2);
```

Each observation is assigned to the validation subset once (and
only once). Thus, a union over all validation index-vectors
reproduces the full range `1:n`. Note that there is no random
assignment of observations to subsets, which means that adjacent
observations are likely to be part of the same validation subset.

```julia-repl
julia> train_idx
5-element Vector{Vector{Int64}}:
 [3,4,5,6,7,8,9,10]
 [1,2,5,6,7,8,9,10]
 [1,2,3,4,7,8,9,10]
 [1,2,3,4,5,6,9,10]
 [1,2,3,4,5,6,7,8]

julia> val_idx
5-element Vector{UnitRange{Int64}}:
 1:2
 3:4
 5:6
 7:8
 9:10
```
"""
function leavepout(n::Integer, p::Integer=1)
    1 <= p <= floor(n / 2) || throw(ArgumentError("p must to be within 1:$(floor(Int,n/2))"))
    k = floor(Int, n / p)
    kfolds(n, k)
end

"""
    leavepout(data, p = 1)

Repartition a `data` container using a k-fold strategy, where `k`
is chosen in such a way, that each validation subset of the
resulting folds contains roughly `p` observations. Defaults to
`p = 1`, which is also known as "leave-one-out" partitioning.

The resulting sequence of folds is returned as a lazy
iterator. Only data subsets are created. That means no actual
data is copied until [`getobs`](@ref) is invoked.

```julia
for (train, val) in leavepout(X, p=2)
    # if numobs(X) is dividable by 2,
    # then numobs(val) will be 2 for each iteraton,
    # otherwise it may be 3 for the first few iterations.
end
```

See[`kfolds`](@ref) for a related function.
"""
function leavepout(data, p::Integer)
    n = numobs(data)
    1 <= p <= floor(n / 2) || throw(ArgumentError("p must to be within 1:$(floor(Int,n/2))"))
    k = floor(Int, n / p)
    kfolds(data, k)
end

leavepout(data; p::Integer=1) = leavepout(data, p)

"""
    timeseries_kfolds(n::Integer; k=5, gap=0, max_train_size=nothing) -> Tuple

Compute the train/validation assignments for `k` time-ordered folds of `n`
observations, and return them in the form of two vectors of index ranges. The
first vector contains the training ranges and the second the validation ranges.

Unlike [`kfolds`](@ref), the validation block of each fold always comes *after*
its training block, so temporal order is respected. This is the equivalent of
scikit-learn's `TimeSeriesSplit` and MLJ's `TimeSeriesCV`.

The last `k * (n ÷ (k+1))` observations are split into `k` contiguous validation
blocks; the training set for each fold is, by default, all earlier observations
(an expanding window). Any observations left over from the non-even division are
absorbed into the first training block.

# Keyword arguments
- `k`: number of folds (default `5`).
- `gap`: number of observations to discard between each training block and its
  validation block (default `0`).
- `max_train_size`: if given, cap each training window to this many of the most
  recent observations (a sliding window instead of an expanding one).

# Examples

```jldoctest
julia> train_idx, val_idx = timeseries_kfolds(10; k=3);

julia> train_idx
3-element Vector{UnitRange{Int64}}:
 1:4
 1:6
 1:8

julia> val_idx
3-element Vector{UnitRange{Int64}}:
 5:6
 7:8
 9:10
```
"""
function timeseries_kfolds(n::Integer; k::Integer=5, gap::Integer=0,
                           max_train_size::Union{Nothing,Integer}=nothing)
    2 <= k <= n - 1 || throw(ArgumentError("k must be within 2:$(max(2, n - 1))"))
    gap >= 0 || throw(ArgumentError("gap must be non-negative"))
    max_train_size === nothing || max_train_size >= 1 ||
        throw(ArgumentError("max_train_size must be positive"))
    fold_size = n ÷ (k + 1)
    fold_size >= 1 || throw(ArgumentError("not enough observations ($n) for k=$k folds"))

    val_indices = Vector{UnitRange{Int}}(undef, k)
    train_indices = Vector{UnitRange{Int}}(undef, k)
    for i in 1:k
        vstart = n - (k - i + 1) * fold_size + 1
        vstop = n - (k - i) * fold_size
        tstop = vstart - 1 - gap
        tstop >= 1 || throw(ArgumentError("gap=$gap is too large for k=$k folds"))
        tstart = max_train_size === nothing ? 1 : max(1, tstop - max_train_size + 1)
        train_indices[i] = tstart:tstop
        val_indices[i] = vstart:vstop
    end
    return train_indices, val_indices
end

"""
    timeseries_kfolds(data; k=5, gap=0, max_train_size=nothing)

Repartition a `data` container using a time-series k-folds strategy and return
the sequence of folds as a lazy iterator. Only data subsets are created, which
means that no actual data is copied until [`getobs`](@ref) is invoked.

Observations are assumed to be in chronological order, i.e. `getobs(data, i)`
precedes `getobs(data, i+1)` in time. The data is *not* sorted, and unlike
[`kfolds`](@ref) it must **not** be shuffled, since that would destroy the
temporal ordering the folds rely on.

```julia
for (train, val) in timeseries_kfolds(X; k=10)
    # the observations in `train` all precede those in `val`
end
```

See the integer method [`timeseries_kfolds(n::Integer)`](@ref) for the keyword
arguments and the precise splitting scheme, and [`kfolds`](@ref) for the
non-temporal variant.
"""
function timeseries_kfolds(data; kws...)
    train_indices, val_indices = timeseries_kfolds(numobs(data); kws...)
    ((obsview(data, itrain), obsview(data, ival))
     for (itrain, ival) in zip(train_indices, val_indices))
end
