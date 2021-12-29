"""
    eachobs(data, buffer=false)

Iterate over `data` one observation at a time. 

If  `buffer=true` and supported by the type of `data`, 
a buffer will be preallocated and reused for memory efficiency.
A buffer object `b` can also be passed directly with `buffer=b`.

```julia
X = rand(4,100)
for x in eachobs(X)
    # loop entered 100 times
    @assert typeof(x) <: Vector{Float64}
    @assert size(x) == (4,)
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachobs((X,Y))
    # ...
end
```

See [`eachbatch`](@ref) for a mini-batch version.
"""
function eachobs(data; buffer=false)
    if buffer === false 
        return (getobs(data, i) for i in 1:numobs(data))
    else
        if buffer === true && numobs(data) > 0
            buffer = getobs(data, 1)
        end
        return (getobs!(buffer, data, i) for i in 1:numobs(data))
    end
end


# --------------------------------------------------------------------

"""
    eachbatch(data; size=1, partial=true, buffer=false)
    
Iterate over `data` one batch at a time. If supported by the type
of `data`, a buffer will be preallocated and reused for memory
efficiency.


The  batch-size can be either provided directly using
`size`. 

```julia
X = rand(4,150)
for x in eachbatch(X, size=10)
    # loop entered 15 times
    @assert typeof(x) <: Matrix{Float64}
    @assert size(x) == (4,10)
end
```

Multiple variables are supported (e.g. for labeled data):

```julia
for (x,y) in eachbatch((X,Y))
    # ...
end
```

See also [`batchview`](@ref) and [`eachobs`](@ref).
"""
function eachbatch(data; buffer=false, kws...)
    batched = BatchView(data; kws...)
    return eachobs(batched; buffer)
end

eachbatch(data, size; kws...) = eachbatch(data; size, kws...)

