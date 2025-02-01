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
  May want to set `partial=false` to avoid size mismatch. 
  Finally, can pass an external buffer to be used in `getobs!`
  (depending on the `collate` and `batchsize` options, could be `getobs!(buffer, data, idxs)` or `getobs!(buffer[i], data, idx)`).
  Default `false`. 
- **`collate`**: Defines the batching behavior. Default `nothing`. 
  - If `nothing` , a batch is `getobs(data, indices)`. 
  - If `false`, each batch is `[getobs(data, i) for i in indices]`. 
  - If `true`, applies `MLUtils.batch` to the vector of observations in a batch, 
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

julia> collate_fn(batch) = join(batch);

julia> first(DataLoader(["a", "b", "c", "d"], batchsize=2, collate=collate_fn))
"ab"
```
"""
struct DataLoader{T<:Union{ObsView,BatchView},B,P,C,O,R<:AbstractRNG}
    data::O  # original data
    _data::T # data wrapped in ObsView / BatchView
    batchsize::Int
    buffer::B    # boolean, or external buffer
    partial::Bool
    shuffle::Bool
    parallel::Bool
    collate::C
    rng::R
end

function DataLoader(
        data;
        buffer = false,
        parallel::Bool = false,
        shuffle::Bool = false,
        batchsize::Int = 1,
        partial::Bool = true,
        collate = Val(nothing),
        rng::AbstractRNG = Random.default_rng())

    if collate isa Bool || collate === nothing
        collate = Val(collate)
    end

    # Wrapping with ObsView in order to work around
    # issue https://github.com/FluxML/Flux.jl/issues/1935    
    _data = ObsView(data)
    if batchsize > 0
        _data = BatchView(_data; batchsize, partial, collate)
    end

    if buffer == true  
        buffer = _create_buffer(_data)
    end
    P = parallel ? :parallel : :serial
    # for buffer == false and external buffer, we keep as is

    T, O, B, C, R = typeof(_data), typeof(data), typeof(buffer), typeof(collate), typeof(rng)
    return DataLoader{T,B,P,C,O,R}(data, _data, batchsize, buffer, 
                                partial, shuffle, parallel, collate, rng) 
end



# buffered - serial case
function Base.iterate(d::DataLoader{T,B,:serial}) where {T,B}
    @assert d.buffer != false 
    data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
    iter = (getobs!(d.buffer, data, i) for i in 1:numobs(data))
    obs, state = iterate(iter)
    return obs, (iter, state)
end

# buffered - parallel case
function Base.iterate(d::DataLoader{T,B,:parallel}) where {T,B}
    @assert d.buffer != false
    data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
    iter = _eachobsparallel_buffered(d.buffer, data)
    obs, state = iterate(iter)
    return obs, (iter, state)
end

# unbuffered - serial case
function Base.iterate(d::DataLoader{T,Bool,:serial}) where {T}
    @assert d.buffer == false 
    data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
    iter = (getobs(data, i) for i in 1:numobs(data))
    obs, state = iterate(iter)
    return obs, (iter, state)
end

# unbuffered - parallel case
function Base.iterate(d::DataLoader{T,Bool,:parallel}) where {T}
    @assert d.buffer == false
    data = d.shuffle ? _shuffledata(d.rng, d._data) : d._data
    iter = _eachobsparallel_unbuffered(data)
    obs, state = iterate(iter)
    return obs, (iter, state)
end

## next iterations
function Base.iterate(::DataLoader, (iter, state))
    ret = iterate(iter, state)
    isnothing(ret) && return
    obs, state = ret
    return obs, (iter, state)
end

_shuffledata(rng, data::ObsView) = shuffleobs(rng, data)

_shuffledata(rng, data::BatchView) = 
    BatchView(shuffleobs(rng, data.data); data.batchsize, data.partial, data.collate)

_create_buffer(x) = getobs(x, 1)

function _create_buffer(x::BatchView)
    obsindices = _batchrange(x, 1)
    return [getobs(A.data, idx) for idx in enumerate(obsindices)]
end

function _create_buffer(x::BatchView{TElem,TData,Val{nothing}}) where {TElem,TData}
    obsindices = _batchrange(x, 1)
    return getobs(x.data, obsindices)
end

Base.length(d::DataLoader) = numobs(d._data)
Base.size(d::DataLoader) = (length(d),)
Base.IteratorEltype(d::DataLoader) = Base.EltypeUnknown()

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

    return DataLoader(d.data;
                    batchsize=d.batchsize,
                    buffer=d.buffer,
                    partial=d.partial,
                    shuffle=d.shuffle,
                    parallel=d.parallel,
                    collate=collate,
                    rng=d.rng)
end


# Base uses this function for composable array printing, e.g. adjoint(view(::Matrix)))
function Base.showarg(io::IO, d::DataLoader, toplevel)
    print(io, "DataLoader(")
    Base.showarg(io, d.data, false)
    d.buffer == false || print(io, ", buffer=", d.buffer)
    d.parallel == false || print(io, ", parallel=", d.parallel)
    d.shuffle == false || print(io, ", shuffle=", d.shuffle)
    d.batchsize == 1 || print(io, ", batchsize=", d.batchsize)
    d.partial == true || print(io, ", partial=", d.partial)
    d.collate === Val(nothing) || print(io, ", collate=", d.collate)
    d.rng == Random.default_rng() || print(io, ", rng=", d.rng)
    print(io, ")")
end

Base.show(io::IO, e::DataLoader) = Base.showarg(io, e, false)

function Base.show(io::IO, m::MIME"text/plain", d::DataLoader)
    print(io, length(d), "-element ")
    Base.showarg(io, d, false)
    print(io, "\n  with first element:")
    print(io, "\n  ", _expanded_summary(first(d)))
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


### TRANSDUCERS IMPLEMENTATION #############################


@inline function _dataloader_foldl1(rf, val, d::DataLoader, data)
    if d.shuffle
        return _dataloader_foldl2(rf, val, d, _shuffledata(d.rng, data))
    else
        return _dataloader_foldl2(rf, val, d, data)
    end
end

@inline function _dataloader_foldl2(rf, val, d::DataLoader, data)
    if d.buffer == false
        return _dataloader_foldl3(rf, val, data)
    else
        return _dataloader_foldl3_buffered(rf, val, data, d.buffer)
    end
end

@inline function _dataloader_foldl3(rf, val, data)
    for i in 1:numobs(data)
        @inbounds x = getobs(data, i)
        # TODO: in 1.8 we could @inline this at the callsite,
        #       optimizer seems to be very sensitive to inlining and
        #       quite brittle in its capacity to keep this type stable
        val = Transducers.@next(rf, val, x)
    end
    return Transducers.complete(rf, val)
end

@inline function _dataloader_foldl3_buffered(rf, val, data, buf)
    for i in 1:numobs(data)
        @inbounds x = getobs!(buf, data, i)
        val = Transducers.@next(rf, val, x)
    end
    return Transducers.complete(rf, val)
end

@inline function Transducers.__foldl__(rf, val, d::DataLoader)
    d.parallel && throw(ArgumentError("Transducer fold protocol not supported on parallel data loads"))
    return _dataloader_foldl1(rf, val, d, d._data)
end
