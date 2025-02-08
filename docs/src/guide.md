```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Guide

## Datasets 

### Basic datasets

A dataset in MLUtils.jl is any object `data` that satisfies the following requisites:
1. Contains a certain number of observations, given by `numobs(data)`.
2. Observations can be accessed by index, using `getobs(data, i)`, with `i` in `1:numobs(data)`.

Since [`numobs`](@ref) and [`getobs`](@ref) are natively implemented for basic types like arrays, tuples, named tuples, dictionaries and Tables.jl's tables, you can use them as datasets without any further ado.

For arrays, the convention is that the last dimension is the observation dimension (sometimes also called the batch dimension),

Let's see some examples. We begin with a simple array:

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

Now let's see an example with named tuples. Notice that the number of observations
as to be the same for all fields:

```jldoctest
julia> data = (x = ones(2, 3), y = [1, 2, 3]);

julia> numobs(data)
3

julia> getobs(data, 2)
(x = [1.0, 1.0], y = 2)
```
Finally, let's consider a table:

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

### Custom datasets

If you have a custom dataset type, you can support the MLUtils.jl interface by implementing the `Base.length` and `Base.getindex` functions, since `numobs` and `getobs` fallback to these functions when they are not specifically implemented.

Here is a barebones example of a custom dataset type:
```jldoctest
julia> struct DummyDataset
           length::Int
       end

julia> Base.length(d::DummyDataset) = d.length

julia> function Base.getindex(d::DummyDataset, i::Int)
         1 <= i <= d.length || throw(ArgumentError("Index out of bounds"))
         return 10*i
       end

julia> data = DummyDataset(10)
DummyDataset(10)

julia> numobs(data)
10

julia> getobs(data, 2)
20
```

This is all it takes to make your custom type compatible with functionalities such as the [`DataLoader`](@ref) type and the [`splitobs`](@ref) function.

## Observation Views

It is common in machine learning pipelines to transform or split the observations contained in a dataset. 
In order to avoid unnecessary memory allocations, MLUtils.jl provides the [`obsview`](@ref) function, which creates a view of the observations at the specified indices, without copying the data.

`obsview(data, indices)` can be used with any dataset `data` and a collection of indices `indices`. By default, 
it returns a wrapper type [`ObsView`](@ref), which behaves like a dataset and can be used with any function that accepts datasets. Users can also specify the behavior of `obsview` on their custom types by implementing the `obsview` method for their type. As an example, for array data, `obsview(data, indices)` will return a subarray:

```jldoctest
julia> data = [1 2 3; 4 5 6]
2×3 Matrix{Int64}:
 1  2  3
 4  5  6

julia> obsview([1 2 3; 4 5 6], 1:2)
2×2 view(::Matrix{Int64}, :, 1:2) with eltype Int64:
 1  2
 4  5
```

When working with arrays, it is also possible to use an [`ObsDim`](@ref) object as input to [`obsview`](@ref) to specify the dimension along which the observations are stored. This is useful when the last dimension is not the observation dimension. 

An example of this are 3D arrays used as inputs to recurrent neural networks and transformers,
usually having size `(n_features, n_timesteps, n_samples)`. In the case in which we want to treat the timesteps as observations, we can proceed as follows:

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

julia> getobs(ov, 1)
3×2 Matrix{Int64}:
 1  13
 2  14
 3  15
```



