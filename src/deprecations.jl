# Deprecated in v0.2
@deprecate unstack(x, dims) unstack(x; dims=dims)
@deprecate unsqueeze(x::AbstractArray, dims::Int) unsqueeze(x; dims=dims)
@deprecate unsqueeze(dims::Int) unsqueeze(dims=dims)
@deprecate labelmap(x) group_indices(x)
@deprecate frequencies(x) group_counts(x)
@deprecate eachbatch(data, batchsize; kws...) eachobs(data; batchsize, kws...)
@deprecate eachbatch(data; size=1, kws...) eachobs(data; batchsize=size, kws...)

# Deprecated in v0.3
import Base: rpad
@deprecate rpad(v::AbstractVector, n::Integer, p) rpad_constant(v, n, p)
