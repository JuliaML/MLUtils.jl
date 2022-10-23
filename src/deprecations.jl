# Deprecated in v0.2
@deprecate stack(x, dims) stack(x; dims=dims)
@deprecate unstack(x, dims) unstack(x; dims=dims)
@deprecate unsqueeze(x::AbstractArray, dims::Int) unsqueeze(x; dims=dims)
@deprecate unsqueeze(dims::Int) unsqueeze(dims=dims)
@deprecate labelmap(x) group_indices(x)
@deprecate frequencies(x) group_counts(x)
@deprecate eachbatch(data, batchsize; kws...) eachobs(data; batchsize, kws...)
@deprecate eachbatch(data; size=1, kws...) eachobs(data; batchsize=size, kws...)

# Deprecated in v0.3

function Base.rpad(v::AbstractVector, n::Integer, p)
    @warn "rpad is deprecated, NNlib.pad_zeros or NNlib.pad_constant instead"
    return [v; fill(p, max(n - length(v), 0))]
end
