# Deprecations v0.1
@deprecate stack(x, dims) stack(x; dims=dims)
@deprecate unstack(x, dims) unstack(x; dims=dims)
@deprecate unsqueeze(x::AbstractArray, dims::Int) unsqueeze(x; dims=dims)
@deprecate unsqueeze(dims::Int) unsqueeze(dims=dims)