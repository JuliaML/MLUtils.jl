
"""
    eachobs(data; [batchsize, buffer, partial, shuffle, rng])

Return an iterator over the observations in the dataset `data`.

Takes as input a single data array, a tuple (or a named tuple) of arrays,
or in general any `data` object that implements the [`numobs`](@ref) and [`getobs`](@ref)
interface (which falls back to `length` and `getindex`).

If `batchsize > 0` the iterations will yield mini-batches of observations.
The last dimension in each array is the observation dimension, i.e. the one
divided into mini-batches.

# Arguments

- `data`: The data to be iterated over. The data type has to be supported by
[`numobs`](@ref) and [`getobs`](@ref).
- `buffer`: If `buffer=true` and supported by the type of `data`,
a buffer will be allocated and reused for memory efficiency.
You can also pass a preallocated object to `buffer`. Default `false`.
- `batchsize`: If less than 0, iterates over individual observations.
Otherwise, each iteration (except possibly the last) yields a mini-batch
containing `batchsize` observations. Default `-1`.
- `partial`: This argument is used only when `batchsize > 0`.
If `partial=false` and the number of observations is not divisible by the batchsize,
then the last mini-batch is dropped. Default `true`.
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

julia> for x in eachobs(Xtrain, batchsize=2)
           @assert size(x) == (10, 2)
           # do something with x, 50 times
       end

julia> Ytrain = rand('a':'z', 100);  # now make a EachObs yielding 2-element named tuples

julia> train_loader = eachobs((data=Xtrain, label=Ytrain), batchsize=5, shuffle=true);

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

julia> foreach(println∘summary, eachobs(rand(Int8, 10, 64), batchsize=30))  # partial=false would omit last
10×30 Matrix{Int8}
10×30 Matrix{Int8}
10×4 Matrix{Int8}
```
"""
function eachobs(data; 
            buffer = false,
            shuffle = false,
            batchsize::Int = -1,
            partial::Bool = true,
            collate = Val(nothing),
            rng::AbstractRNG = Random.GLOBAL_RNG)
    buffer = buffer isa Bool ? buffer : true
    collate = collate isa Val ? collate : Val(collate)
    if !(collate ∈ (Val(nothing), Val(true), Val(false)))
        throw(ArgumentError("`collate` must be one of `nothing`, `true` or `false`."))
    end
    return EachObs(data, batchsize, buffer, partial, shuffle, collate, rng)
end

struct EachObs{T, R<:AbstractRNG, C<:Val}
    data::T
    batchsize::Int
    buffer::Bool
    partial::Bool
    shuffle::Bool
    collate::C
    rng::R
end

function Base.iterate(e::EachObs)
    # Wrapping with ObsView in order to work around
    # issue https://github.com/FluxML/Flux.jl/issues/1935
    data = ObsView(e.data)

    data = e.shuffle ? shuffleobs(e.rng, data) : data
    data = e.batchsize > 0 ? BatchView(data; e.batchsize, e.partial, e.collate) : data

    iter = if e.buffer
                buf = getobs(data, 1)
                (getobs!(buf, data, i) for i in 1:numobs(data))
            else
                (getobs(data, i) for i in 1:numobs(data))
            end

    obs, state = iterate(iter)
    return obs, (iter, state)
end


function Base.iterate(::EachObs, (iter, state))
    ret = iterate(iter, state)
    isnothing(ret) && return
    obs, state = ret
    return obs, (iter, state)
end


function Base.length(e::EachObs)
    numobs(if e.batchsize > 0
        # Wrapping with ObsView in order to work around
        # issue https://github.com/FluxML/Flux.jl/issues/1935
        data = ObsView(e.data)

        BatchView(data; e.batchsize, e.partial)
    else
        e.data
    end)
end


Base.IteratorEltype(::EachObs) = Base.EltypeUnknown()

## This causes error in some cases of `collect(loader)`
# function Base.eltype(e::EachObs)
#     eltype(if e.batchsize > 0
#         BatchView(e.data; e.batchsize, e.partial)
#     else
#         e.data
#     end)
# end
