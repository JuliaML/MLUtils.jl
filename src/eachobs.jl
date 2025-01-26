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
    return DataLoader(data; batchsize, kws...)
end

"""
    DataLoader(data; [batchsize, buffer, collate, parallel, partial, rng, shuffle])

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

- **`data`**: The data to be iterated over. The data type has to be supported by
  [`numobs`](@ref) and [`getobs`](@ref).
- **`batchsize`**: If less than 0, iterates over individual observations.
  Otherwise, each iteration (except possibly the last) yields a mini-batch
  containing `batchsize` observations. Default `1`.
- **`buffer`**: If `buffer=true` and supported by the type of `data`,
  a buffer will be allocated and reused for memory efficiency.
  May want to set `partial=false` to avoid size mismatch. Default `false`. 
- **`collate`**: Defines the batching behavior. Default `nothing`. 
  - If `nothing` , a batch is `getobs(data, indices)`. 
  - If `false`, each batch is `[getobs(data, i) for i in indices]`. 
  - If `true`, applies MLUtils to the vector of observations in a batch, 
    recursively collating arrays in the last dimensions. See [`MLUtils.batch`](@ref) for more information
    and examples.
  - If a custom function, it will be used in place of `MLUtils.batch`. It should take a vector of observations as input.
- **`parallel`**: Whether to use load data in parallel using worker threads. Greatly
    speeds up data loading by factor of available threads. Requires starting
    Julia with multiple threads. Check `Threads.nthreads()` to see the number of
    available threads. **Passing `parallel = true` breaks ordering guarantees**.
    Default `false`.
- **`partial`**: This argument is used only when `batchsize > 0`.
  If `partial=false` and the number of observations is not divisible by the batchsize,
  then the last mini-batch is dropped. Default `true`.
- **`rng`**: A random number generator. Default `Random.default_rng()`.
- **`shuffle**: Whether to shuffle the observations before iterating. Unlike
    wrapping the data container with `shuffleobs(data)`, `shuffle=true` ensures
    that the observations are shuffled anew every time you start iterating over
    `eachobs`. Default `false`.

# Examples

```jldoctest
julia> Xtrain = rand(10, 100);

julia> array_loader = DataLoader(Xtrain, batchsize=2

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


julia> collate_fn(batch) = join(batch)

julia> first(DataLoader(["a", "b", "c", "d"], batchsize=2, collate=collate_fn)) == "ab"
true
```
"""
struct DataLoader{T, R<:AbstractRNG, C}
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
        buffer::Bool = false,
        parallel::Bool = false,
        shuffle::Bool = false,
        batchsize::Int = 1,
        partial::Bool = true,
        collate = Val(nothing),
        rng::AbstractRNG = Random.default_rng())

    if !(buffer isa Bool) && parallel
        throw(ArgumentError("If `parallel=true`, `buffer` must be a boolean."))
    end

    if collate isa Bool || collate === nothing
        collate = Val(collate)
    end
    return DataLoader(data, batchsize, buffer, partial, shuffle, parallel, collate, rng)
end

function Base.iterate(d::DataLoader)
    # TODO move ObsView and BatchWView wrapping to the constructor, so that 
    # we can parametrize the DataLoader with ObsView and BatchView and define specialized methods.
    
    # Wrapping with ObsView in order to work around
    # issue https://github.com/FluxML/Flux.jl/issues/1935
    data = ObsView(d.data)

    data = d.shuffle ? shuffleobs(d.rng, data) : data
    data = d.batchsize > 0 ? BatchView(data; d.batchsize, d.partial, d.collate) : data

    if d.parallel
        iter = eachobsparallel(data; d.buffer)
    else
        if d.buffer == false
            iter = (getobs(data, i) for i in 1:numobs(data))
        elseif d.buffer == true
            buf = getobs(data, 1)
            iter = (getobs!(buf, data, i) for i in 1:numobs(data))
        else # external buffer
            buf = d.buffer
            iter = (getobs!(buf, data, i) for i in 1:numobs(data))
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


function Base.length(d::DataLoader)
    if d.batchsize > 0
        return numobs(BatchView(d.data; d.batchsize, d.partial))
    else
        return numobs(d.data)
    end
end

Base.size(e::DataLoader) = (length(e),)


Base.IteratorEltype(::DataLoader) = Base.EltypeUnknown()

## This causes error in some cases of `collect(loader)`
# function Base.eltype(e::DataLoader)
#     eltype(if e.batchsize > 0
#         BatchView(e.data; e.batchsize, e.partial)
#     else
#         e.data
#     end)
# end

"""
    mapobs(f, d::DataLoader)

Return a new dataloader based on `d`  that applies `f` at each iteration. 

# Examples

```jldoctest
julia> X = ones(3, 6);

julia> function f(x)
           @show x
           return x
       end
f (generic function with 1 method)

julia> d = DataLoader(X, batchsize=2, collate=false);

julia> d = mapobs(f, d);

julia> for x in d
           @assert size(x) == (2,)
           @assert size(x[1]) == (3,)
       end
x = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
x = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
x = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

julia> d2 = DataLoader(X, batchsize=2, collate=true);

julia> d2 = mapobs(f, d2);

julia> for x in d2
           @assert size(x) == (3, 2)
       end
x = [1.0 1.0; 1.0 1.0; 1.0 1.0]
x = [1.0 1.0; 1.0 1.0; 1.0 1.0]
x = [1.0 1.0; 1.0 1.0; 1.0 1.0]
```
"""
function mapobs(f, d::DataLoader)
    @assert d.batchsize > 0 "Mapping over individual observations is not supported, set `batchsize > 0` 
    or apply mapobs to the underlying data container."
   
    @assert d.collate !== Val(nothing) "`collate==nothing` not supported by mapobs"
    if d.collate == Val(false)
        collate = f
    elseif d.collate === Val(true)
        collate = f ∘ batch
    else
        collate = f ∘ d.collate
    end

    DataLoader(d.data,
               batchsize=d.batchsize,
               buffer=d.buffer,
               partial=d.partial,
               shuffle=d.shuffle,
               parallel=d.parallel,
               collate=collate,
               rng=d.rng)
end


@inline function _dataloader_foldl1(rf, val, e::DataLoader, data)
    if e.shuffle
        _dataloader_foldl2(rf, val, e, shuffleobs(e.rng, data))
    else
        _dataloader_foldl2(rf, val, e, data)
    end
end

@inline function _dataloader_foldl2(rf, val, e::DataLoader, data)
    if e.batchsize > 0
        _dataloader_foldl3(rf, val, e, BatchView(data; e.batchsize, e.partial))
    else
        _dataloader_foldl3(rf, val, e, data)
    end
end

@inline function _dataloader_foldl3(rf, val, e::DataLoader, data)
    if e.buffer > 0
        _dataloader_foldl4_buffered(rf, val, data)
    else
        _dataloader_foldl4(rf, val, data)
    end
end

@inline function _dataloader_foldl4(rf, val, data)
    for i in 1:numobs(data)
        @inbounds x = getobs(data, i)
        # TODO: in 1.8 we could @inline this at the callsite,
        #       optimizer seems to be very sensitive to inlining and
        #       quite brittle in its capacity to keep this type stable
        val = Transducers.@next(rf, val, x)
    end
    Transducers.complete(rf, val)
end

@inline function _dataloader_foldl4_buffered(rf, val, data)
    buf = getobs(data, 1)
    for i in 1:numobs(data)
        @inbounds x = getobs!(buf, data, i)
        val = Transducers.@next(rf, val, x)
    end
    Transducers.complete(rf, val)
end

@inline function Transducers.__foldl__(rf, val, e::DataLoader)
    e.parallel && throw(ArgumentError("Transducer fold protocol not supported on parallel data loads"))
    _dataloader_foldl1(rf, val, e, ObsView(e.data))
end

# Base uses this function for composable array printing, e.g. adjoint(view(::Matrix)))
function Base.showarg(io::IO, e::DataLoader, toplevel)
    print(io, "DataLoader(")
    Base.showarg(io, e.data, false)
    e.buffer == false || print(io, ", buffer=", e.buffer)
    e.parallel == false || print(io, ", parallel=", e.parallel)
    e.shuffle == false || print(io, ", shuffle=", e.shuffle)
    e.batchsize == 1 || print(io, ", batchsize=", e.batchsize)
    e.partial == true || print(io, ", partial=", e.partial)
    e.collate === Val(nothing) || print(io, ", collate=", e.collate)
    e.rng == Random.default_rng() || print(io, ", rng=", e.rng)
    print(io, ")")
end

Base.show(io::IO, e::DataLoader) = Base.showarg(io, e, false)

function Base.show(io::IO, m::MIME"text/plain", e::DataLoader)
    if Base.haslength(e)
        print(io, length(e), "-element ")
    else
        print(io, "Unknown-length ")
    end
    Base.showarg(io, e, false)
    print(io, "\n  with first element:")
    print(io, "\n  ", _expanded_summary(first(e)))
end

_expanded_summary(x) = summary(x)
function _expanded_summary(xs::Tuple)
  parts = [_expanded_summary(x) for x in xs]
  "(" * join(parts, ", ") * ",)"
end
function _expanded_summary(xs::NamedTuple)
  parts = ["$k = "*_expanded_summary(x) for (k,x) in zip(keys(xs), xs)]
  "(; " * join(parts, ", ") * ")"
end

