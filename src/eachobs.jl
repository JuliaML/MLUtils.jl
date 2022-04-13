"""
    eachobs(data; kws...)

Return an iterator over `data`.

Supports the same arguments as [`DataLoader`](@ref).
The `batchsize` default is `-1` here while
it is `1` for `DataLoader`.

# Examples

```julia
X = rand(4,100)

for x in eachobs(X)
    # loop entered 100 times
    @assert typeof(x) <: Vector{Float64}
    @assert size(x) == (4,)
end

# mini-batch iterations
for x in eachobs(X, batchsize=10)
    # loop entered 10 times
    @assert typeof(x) <: Matrix{Float64}
    @assert size(x) == (4,10)
end

# support for tuples, named tuples, dicts
for (x, y) in eachobs((X, Y))
    # ...
end
```
"""
function eachobs(data; batchsize=-1, kws...)
    DataLoader(data; batchsize, kws...)
end

"""
    DataLoader(data; [batchsize, buffer, partial, shuffle, parallel, rng])

An object that iterates over mini-batches of `data`,
each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single data array, a tuple (or a named tuple) of arrays,
or in general any `data` object that implements the [`numobs`](@ref) and [`getobs`](@ref)
methods.

The last dimension in each array is the observation dimension, i.e. the one
divided into mini-batches.

The original data is preserved in the `data` field of the DataLoader.

# Arguments

- `data`: The data to be iterated over. The data type has to be supported by
  [`numobs`](@ref) and [`getobs`](@ref).
- `buffer`: If `buffer=true` and supported by the type of `data`,
a buffer will be allocated and reused for memory efficiency.
You can also pass a preallocated object to `buffer`. Default `false`.
- `batchsize`: If less than 0, iterates over individual observations.
Otherwise, each iteration (except possibly the last) yields a mini-batch
containing `batchsize` observations. Default `1`.
- `partial`: This argument is used only when `batchsize > 0`.
  If `partial=false` and the number of observations is not divisible by the batchsize,
  then the last mini-batch is dropped. Default `true`.
- `parallel`: Whether to use load data in parallel using worker threads. Greatly
    speeds up data loading by factor of available threads. Requires starting
    Julia with multiple threads. Check `Threads.nthreads()` to see the number of
    available threads. **Passing `parallel = true` breaks ordering guarantees**.
    Default `false`.
- `shuffle`: Whether to shuffle the observations before iterating. Unlike
    wrapping the data container with `shuffleobs(data)`, `shuffle=true` ensures
    that the observations are shuffled anew every time you start iterating over
    `eachobs`. Default `false`.
- `collate`: Batching behavior. If `nothing` (default), a batch is `getobs(data, indices)`. If `false`, each batch is
    `[getobs(data, i) for i in indices]`. When `true`, applies [`batch`](@ref) to the vector of observations in a batch, 
   recursively collating arrays in the last dimensions. See [`batch`](@ref) for more information and examples.
- `rng`: A random number generator. Default `Random.GLOBAL_RNG`

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
struct DataLoader{T, R<:AbstractRNG, C<:Val}
    data::T
    batchsize::Int
    buffer::Bool
    partial::Bool
    shuffle::Bool
    parallel::Bool
    collate::C
    rng::R
end

function DataLoader(
        data;
        buffer = false,
        parallel = false,
        shuffle = false,
        batchsize::Int = 1,
        partial::Bool = true,
        collate = Val(nothing),
        rng::AbstractRNG = Random.GLOBAL_RNG)
    buffer = buffer isa Bool ? buffer : true
    collate = collate isa Val ? collate : Val(collate)
    if !(collate ∈ (Val(nothing), Val(true), Val(false)))
        throw(ArgumentError("`collate` must be one of `nothing`, `true` or `false`."))
    end
    return DataLoader(data, batchsize, buffer, partial, shuffle, parallel, collate, rng)
end

function Base.iterate(e::DataLoader)
    # Wrapping with ObsView in order to work around
    # issue https://github.com/FluxML/Flux.jl/issues/1935
    data = ObsView(e.data)

    data = e.shuffle ? shuffleobs(e.rng, data) : data
    data = e.batchsize > 0 ? BatchView(data; e.batchsize, e.partial, e.collate) : data

    iter = if e.parallel
        eachobsparallel(data; e.buffer)
    else
        if e.buffer
            buf = getobs(data, 1)
            (getobs!(buf, data, i) for i in 1:numobs(data))
        else
            (getobs(data, i) for i in 1:numobs(data))
        end
    end
    obs, state = iterate(iter)
    return obs, (iter, state)
end


function Base.iterate(::DataLoader, (iter, state))
    ret = iterate(iter, state)
    isnothing(ret) && return
    obs, state = ret
    return obs, (iter, state)
end


function Base.length(e::DataLoader)
    numobs(if e.batchsize > 0
        # Wrapping with ObsView in order to work around
        # issue https://github.com/FluxML/Flux.jl/issues/1935
        data = ObsView(e.data)

        BatchView(data; e.batchsize, e.partial)
    else
        e.data
    end)
end


Base.IteratorEltype(::DataLoader) = Base.EltypeUnknown()

## This causes error in some cases of `collect(loader)`
# function Base.eltype(e::DataLoader)
#     eltype(if e.batchsize > 0
#         BatchView(e.data; e.batchsize, e.partial)
#     else
#         e.data
#     end)
# end

function _dataloader_foldl(rf, val, data)
    for i in 1:numobs(data)
        @inbounds x = getobs(data, i)
        # TODO: in 1.8 we could @inline this at the callsite,
        #       optimizer seems to be very sensitive to inlining and
        #       quite brittle in its capacity to keep this type stable
        val = Transducers.@next(rf, val, x)
    end
    Transducers.complete(rf, val)
end

function _dataloader_foldl_buffered(rf, val, data)
    buf = getobs(data, 1)
    for i in 1:numobs(data)
        @inbounds x = getobs!(buf, data, i)
        # TODO: in 1.8 we could @inline this at the callsite,
        #       optimizer seems to be very sensitive to inlining and
        #       quite brittle in its capacity to keep this type stable
        val = Transducers.@next(rf, val, x)
    end
    Transducers.complete(rf, val)
end

function Transducers.__foldl__(rf, val, e::DataLoader)
    e.parallel && throw(ArgumentError("Transducer fold protocol not supported on parallel data loads"))
    data = ObsView(e.data)
    data = e.shuffle ? shuffleobs(e.rng, data) : data
    data = e.batchsize > 0 ? BatchView(data; e.batchsize, e.partial, e.collate) : data

    # Indirect to type stabilize `data`
    if e.buffer
        _dataloader_foldl_buffered(rf, val, data)
    else
        _dataloader_foldl(rf, val, data)
    end
end