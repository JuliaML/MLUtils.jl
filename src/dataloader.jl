# Adapted from Knet's src/data.jl (author: Deniz Yuret)

struct DataLoader{D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    shuffle::Bool
    rng::R
end

"""
    DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)

An object that iterates over mini-batches of `data`, 
each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single data tensor, or a tuple (or a named tuple) of tensors.
The last dimension in each tensor is the observation dimension, i.e. the one
divided into mini-batches.

If `shuffle=true`, it shuffles the observations each time iterations are re-started.
If `partial=false` and the number of observations is not divisible by the batchsize, 
then the last mini-batch is dropped.

The original data is preserved in the `data` field of the DataLoader.

# Examples

```jldoctest
julia> Xtrain = rand(10, 100);

julia> array_loader = DataLoader(Xtrain, batchsize=2);

julia> for x in array_loader
         @assert size(x) == (10, 2)
         # do something with x, 50 times
       end

julia> array_loader.data === Xtrain
true

julia> tuple_loader = DataLoader((Xtrain,), batchsize=2);  # similar, but yielding 1-element tuples

julia> for x in tuple_loader
         @assert x isa Tuple{Matrix}
         @assert size(x[1]) == (10, 2)
       end

julia> Ytrain = rand('a':'z', 100);  # now make a DataLoader yielding 2-element named tuples

julia> train_loader = DataLoader((data=Xtrain, label=Ytrain), batchsize=5, shuffle=true);

julia> for epoch in 1:100
         for (x, y) in train_loader  # access via tuple destructuring
           @assert size(x) == (10, 5)
           @assert size(y) == (5,)
           # loss += f(x, y) # etc, runs 100 * 20 times
         end
       end

julia> first(train_loader).label isa Vector{Char}  # access via property name
true

julia> first(train_loader).label == Ytrain[1:5]  # because of shuffle=true
false

julia> foreach(println∘summary, DataLoader(rand(Int8, 10, 64), batchsize=30))  # partial=false would omit last
10×30 Matrix{Int8}
10×30 Matrix{Int8}
10×4 Matrix{Int8}
```
"""
function DataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))
    nobs = numobs(data)
    if nobs < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $nobs"
        batchsize = nobs
    end
    DataLoader(data, batchsize, nobs, partial, shuffle, rng)
end

@propagate_inbounds function Base.iterate(d::DataLoader)
    data = d.data
    if d.shuffle
        data = shuffleobs(d.rng, data)
    end
    gen = eachobs(data; d.batchsize, d.partial)
    res = iterate(gen)
    res === nothing && return
    res[1], (gen, res[2])
end

@propagate_inbounds function Base.iterate(d::DataLoader, state)     
    gen, i = state
    res = iterate(gen, i)
    res === nothing && return
    res[1], (gen, res[2])
end

function Base.length(d::DataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int, n) : floor(Int, n)
end

const BasicDatasets = Union{AbstractArray, Tuple, NamedTuple, Dict}

Base.IteratorEltype(d::DataLoader) = Base.EltypeUnknown()
Base.IteratorEltype(d::DataLoader{<:BasicDatasets}) = Base.HasEltype()

Base.eltype(::Type{<:DataLoader{D}}) where D = datatype(D)

datatype(D::Type) = Any
datatype(D::Type{<:AbstractArray}) = D
datatype(D::Type{<:Tuple}) = datatype.(D)
datatype(D::Type{<:NamedTuple}) = datatype.(D)
datatype(D::Type{Dict{K, V}}) where {K,V} = Dict{K, datatype(V)}
