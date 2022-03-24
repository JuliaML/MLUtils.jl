# Deprecations v0.1
@deprecate stack(x, dims) stack(x; dims=dims)
@deprecate unstack(x, dims) unstack(x; dims=dims)
@deprecate unsqueeze(x::AbstractArray, dims::Int) unsqueeze(x; dims=dims)
@deprecate unsqueeze(dims::Int) unsqueeze(dims=dims)
@deprecate labelmap(x) group_indices(x)
@deprecate frequencies(x) group_counts(x)
@deprecate eachbatch(data, batchsize; kws...) eachobs(data; batchsize, kws...)
@deprecate eachbatch(data; size=1, kws...) eachobs(data; batchsize=size, kws...)
