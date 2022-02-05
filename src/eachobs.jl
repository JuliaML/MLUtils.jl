"""
    eachobs(data, buffer=false, batchsize=-1, partial=true)

Return an iterator over the observations in `data`. 

# Arguments

- `data`. The data to be iterated over. The data type has to implement 
  [`numobs`](@ref) and [`getobs`](@ref). 
- `buffer`. If `buffer=true` and supported by the type of `data`, 
a buffer will be allocated and reused for memory efficiency.
You can also pass a preallocated object to `buffer`.
- `batchsize`. If less than 0, iterates over individual observation. 
Otherwise, each iteration (except possibly the last) yields a mini-batch 
containing `batchsize` observations.
- `partial`. This argument is used only when `batchsize > 0`.
  If `partial=false` and the number of observations is not divisible by the batchsize, 
  then the last mini-batch is dropped.

See also [`numobs`](@ref), [`getobs`](@ref).

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
function eachobs(data; buffer = false, batchsize::Int = -1, partial::Bool =true)
    if batchsize > 0
        data = BatchView(data; batchsize, partial)
    end
    if buffer === false 
        gen = (getobs(data, i) for i in 1:numobs(data))
    else
        if buffer === true && numobs(data) > 0
            buffer = getobs(data, 1)
        end
        gen = (getobs!(buffer, data, i) for i in 1:numobs(data))
    end
    return gen
end

