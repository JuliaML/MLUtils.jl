"""
    oversample([rng], data, classes; fraction=1, shuffle=true)
    oversample([rng], data::Tuple; fraction=1, shuffle=true)

Generate a re-balanced version of `data` by repeatedly sampling
existing observations in such a way that every class will have at
least `fraction` times the number observations of the largest
class in `classes`. This way, all classes will have a minimum number of
observations in the resulting data set relative to what largest
class has in the given (original) `data`.

As an example, by default (i.e. with `fraction = 1`) the
resulting dataset will be near perfectly balanced. On the other
hand, with `fraction = 0.5` every class in the resulting data
with have at least 50% as many observations as the largest class.

The `classes` input is an array with the same length as `numobs(data)`.  

The convenience parameter `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the repeated samples will be together at the
end, sorted by class. Defaults to `true`.

The random number generator `rng` can be optionally passed as the
first argument. 

The output will contain both the resampled data and classes.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# oversample the class "a" to match "b"
X_bal, Y_bal = oversample(X, Y)

# this results in a bigger dataset with repeated data
@assert size(X_bal) == (3,8)
@assert length(Y_bal) == 8

# now both "a", and "b" have 4 observations each
@assert sum(Y_bal .== "a") == 4
@assert sum(Y_bal .== "b") == 4
```

For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref). 

If `data` is a tuple and `classes` is not given, 
then it will be assumed that the last element of the tuple contains the classes.

```julia
julia> data = DataFrame(X1=rand(6), X2=rand(6), Y=[:a,:b,:b,:b,:b,:a])
6×3 DataFrames.DataFrame
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.226582  │ 0.0443222   │ a │
│ 2   │ 0.504629  │ 0.722906    │ b │
│ 3   │ 0.933372  │ 0.812814    │ b │
│ 4   │ 0.522172  │ 0.245457    │ b │
│ 5   │ 0.505208  │ 0.11202     │ b │
│ 6   │ 0.0997825 │ 0.000341996 │ a │

julia> getobs(oversample(data, data.Y))
8×3 DataFrame
 Row │ X1        X2         Y      
     │ Float64   Float64    Symbol 
─────┼─────────────────────────────
   1 │ 0.376304  0.100022   a
   2 │ 0.467095  0.185437   b
   3 │ 0.481957  0.319906   b
   4 │ 0.336762  0.390811   b
   5 │ 0.376304  0.100022   a
   6 │ 0.427064  0.0648339  a
   7 │ 0.427064  0.0648339  a
   8 │ 0.457043  0.490688   b
```

See [`ObsView`](@ref) for more information on data subsets.
See also [`undersample`](@ref).
"""
oversample(data, classes; kws...) = oversample(Random.default_rng(), data, classes; kws...)
oversample(data::Tuple; kws...) = oversample(Random.default_rng(), data; kws...)

function oversample(rng::AbstractRNG, data, classes; fraction=1, shuffle::Bool=true)
    lm = group_indices(classes)

    maxcount = maximum(length, values(lm))
    fraccount = round(Int, fraction * maxcount)

    # firstly we will start by keeping everything
    inds = collect(1:numobs(data))
    
    for (lbl, inds_for_lbl) in lm
        num_extra_needed = fraccount - length(inds_for_lbl)
        while num_extra_needed > length(inds_for_lbl)
            num_extra_needed -= length(inds_for_lbl)
            append!(inds, inds_for_lbl)
        end
        if num_extra_needed > 0
            if shuffle
                append!(inds, sample(rng, inds_for_lbl, num_extra_needed; replace=false))
            else
                append!(inds, inds_for_lbl[1:num_extra_needed])
            end
        end
    end

    shuffle && shuffle!(rng, inds)
    return obsview(data, inds), obsview(classes, inds)
end

function oversample(rng::AbstractRNG, data::Tuple; kws...)
    d, c = oversample(rng, data[1:end-1], data[end]; kws...)
    return (d..., c)
end

"""
    undersample([rng], data, classes; shuffle=true)
    undersample([rng], data::Tuple; shuffle=true)

Generate a class-balanced version of `data` by subsampling its
observations in such a way that the resulting number of
observations will be the same number for every class. This way,
all classes will have as many observations in the resulting data
set as the smallest class has in the given (original) `data`.

The convenience parameter `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the observations will be in their original
order. Defaults to `false`.

If `data` is a tuple and `classes` is not given, 
then it will be assumed that the last element of the tuple contains the classes.

The output will contain both the resampled data and classes.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# subsample the class "b" to match "a"
X_bal, Y_bal = undersample(X, Y)

# this results in a smaller dataset
@assert size(X_bal) == (3,4)
@assert length(Y_bal) == 4

# now both "a", and "b" have 2 observations each
@assert sum(Y_bal .== "a") == 2
@assert sum(Y_bal .== "b") == 2
```

For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref). 

Note that if `data` is a tuple, then it will be assumed that the
last element of the tuple contains the targets.

```julia
julia> data = DataFrame(X1=rand(6), X2=rand(6), Y=[:a,:b,:b,:b,:b,:a])
6×3 DataFrames.DataFrame
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.226582  │ 0.0443222   │ a │
│ 2   │ 0.504629  │ 0.722906    │ b │
│ 3   │ 0.933372  │ 0.812814    │ b │
│ 4   │ 0.522172  │ 0.245457    │ b │
│ 5   │ 0.505208  │ 0.11202     │ b │
│ 6   │ 0.0997825 │ 0.000341996 │ a │

julia> getobs(undersample(data, data.Y))
4×3 DataFrame
 Row │ X1        X2         Y      
     │ Float64   Float64    Symbol 
─────┼─────────────────────────────
   1 │ 0.427064  0.0648339  a
   2 │ 0.376304  0.100022   a
   3 │ 0.467095  0.185437   b
   4 │ 0.457043  0.490688   b
```

See [`ObsView`](@ref) for more information on data subsets.
See also [`oversample`](@ref).
"""
undersample(data, classes; kws...) = undersample(Random.default_rng(), data, classes; kws...)
undersample(data::Tuple; kws...) = undersample(Random.default_rng(), data; kws...)

function undersample(rng::AbstractRNG, data, classes; shuffle::Bool=true)
    lm = group_indices(classes)
    mincount = minimum(length, values(lm))

    inds = Int[]
    
    for (lbl, inds_for_lbl) in lm
        if shuffle
            append!(inds, sample(rng, inds_for_lbl, mincount; replace=false))
        else
            append!(inds, inds_for_lbl[1:mincount])
        end
    end

    shuffle ? shuffle!(rng, inds) : sort!(inds)
    return obsview(data, inds), obsview(classes, inds)
end

function undersample(rng::AbstractRNG, data::Tuple; kws...)
    d, c = undersample(rng, data[1:end-1], data[end]; kws...)
    return (d..., c)
end


"""
    stratifiedobs([rng], data, p; [shuffle = true]) -> Tuple

Partition the dataset `data` into multiple disjoint subsets 
with size proportional to the value(s) of `p`. 
The observations are assignmed to a data subset using stratified sampling without replacement. 

If `p` is a float between 0 and 1, then the return value
will be a tuple with two subsests in which the
first element contains the fraction of observations specified by
`p` and the second element contains the rest. In the following
code the first subset `train` will contain around 70% of the
observations and the second subset `test` the rest. The key
difference to [`splitobs`](@ref) is that the class distribution
in `y` will actively be preserved in `train` and `test`.

```julia
train_data, test_data = stratifiedobs(data, p = 0.7)
```

If `p` is a tuple of floats between 0 and 1, then additional subsets will be
created. In this example `train` will contain about 50% of the
observations, `val` will contain around 30%, and `test` the
remaining 20%.

```julia
train_data, val_data, test_data = stratifiedobs(y, p = (0.5, 0.3))
```

It is also possible to call `stratifiedobs` with multiple data
arguments as tuple, which all must have the same number of total
observations. Note that if `data` is a tuple, then it will be
assumed that the last element of the tuple contains the targets.

```julia
(X_train, y_train), (X_test, y_test) = stratifiedobs((X, y), p = 0.7)
```

The optional parameter `shuffle` determines if the resulting data
subsets should be shuffled. If `false`, then the observations in
the subsets will be grouped together according to their labels.

```julia
julia> y = ["a", "b", "b", "b", "b", "a"] # 2 imbalanced classes
6-element Array{String,1}:
 "a"
 "b"
 "b"
 "b"
 "b"
 "a"

julia> train, test = stratifiedobs(y, p = 0.5, shuffle = false)
(String["b","b","a"],String["b","b","a"])
```

The optional argument `rng` allows one to specify the
random number generator used for shuffling. 

For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref). 

See also [`undersample`](@ref), [`oversample`](@ref), and [`splitobs`](@ref).
"""
function stratifiedobs(data; p = 0.7, shuffle = true, obsdim = default_obsdim(data), rng = Random.GLOBAL_RNG)
    stratifiedobs(identity, data, p, shuffle, convert(ObsDimension, obsdim), rng)
end

function stratifiedobs(f, data; p = 0.7, shuffle = true, obsdim = default_obsdim(data), rng = Random.GLOBAL_RNG)
    stratifiedobs(f, data, p, shuffle, convert(ObsDimension, obsdim), rng)
end

function stratifiedobs(data, p::AbstractFloat, args...)
    stratifiedobs(identity, data, p, args...)
end

function stratifiedobs(data, p::NTuple{N,AbstractFloat}, args...) where N
    stratifiedobs(identity, data, p, args...)
end

function stratifiedobs(rng, data, p::Union{NTuple,AbstractFloat}, stratified::AbstractVector)
    # The given data is always shuffled to qualify as performing
    # stratified sampling without replacement.
    idxs_groups = group_indices(stratified)
    idxs_splits = ntuple(i -> Int[], length(p)+1)
    for (lbl, idxs) in idxs_groups
        new_idxs_splits = splitobs(rng, idxs, at=p)
        for i in 1:length(idxs_splits)
            append!(idxs_splits[i], new_idxs_splits[i])
        end
    end
    return map(idx -> obsview(data, idx), idxs_splits)
end
