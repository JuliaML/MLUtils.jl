"""
    oversample(data, classes; fraction=1, shuffle=true)
    oversample(data::Tuple; fraction=1, shuffle=true)

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
[`numobs`](@ref) and [`getobs`](@ref). For example, the following
code allows `oversample` to work on a `DataFrame`.

```julia
# Make DataFrames.jl work
MLUtils.getobs(data::DataFrame, i) = data[i,:]
MLUtils.numobs(data::DataFrame) = nrow(data)
```

Note that if `data` is a tuple and `classes` is not given, 
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
See also [`undersample`](@ref) and [`stratifiedobs`](@ref).
"""
function oversample(data, classes; fraction=1, shuffle::Bool=true)
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
            append!(inds, sample(inds_for_lbl, num_extra_needed; replace=false))
        end
    end

    shuffle && shuffle!(inds)
    return obsview(data, inds)
end

oversample(data::Tuple; kws...) = oversample(data, data[end]; kws...)


"""
    undersample(data, classes; shuffle=true)

Generate a class-balanced version of `data` by subsampling its
observations in such a way that the resulting number of
observations will be the same number for every class. This way,
all classes will have as many observations in the resulting data
set as the smallest class has in the given (original) `data`.

The convenience parameter `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the observations will be in their original
order. Defaults to `false`.

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
[`numobs`](@ref) and [`getobs`](@ref). For example, the following
code allows `undersample` to work on a `DataFrame`.

```julia
# Make DataFrames.jl work
MLUtils.getobs(data::DataFrame, i) = data[i,:]
MLUtils.numobs(data::DataFrame) = nrow(data)
```
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
See also [`oversample`](@ref) and [`stratifiedobs`](@ref).
"""
function undersample(data, classes; shuffle::Bool=true)
    lm = group_indices(classes)
    mincount = minimum(length, values(lm))

    inds = Int[]
    
    for (lbl, inds_for_lbl) in lm
        append!(inds, sample(inds_for_lbl, mincount; replace=false))
    end

    shuffle ? shuffle!(inds) : sort!(inds)
    return obsview(data, inds)
end

undersample(data::Tuple; kws...) = undersample(data, data[end]; kws...)
