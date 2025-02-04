
"""
    batch(xs)

Batch the arrays in `xs` into a single array with 
an extra dimension.

If the elements of `xs` are tuples, named tuples, or dicts, 
the output will be of the same type. 

See also [`unbatch`](@ref) and [`batch_sequence`](@ref).

# Examples

```jldoctest
julia> batch([[1,2,3], 
              [4,5,6]])
3×2 Matrix{Int64}:
 1  4
 2  5
 3  6

julia> batch([(a=[1,2], b=[3,4])
               (a=[5,6], b=[7,8])]) 
(a = [1 5; 2 6], b = [3 7; 4 8])
```
"""
function batch(xs)
# Fallback for generric iterables
    @assert length(xs) > 0 "Input should be non-empty" 
    data = first(xs) isa AbstractArray ?
        similar(first(xs), size(first(xs))..., length(xs)) :
        Vector{eltype(xs)}(undef, length(xs))
    for (i, x) in enumerate(xs)
        data[batchindex(data, i)...] = x
    end
    return data
end

batchindex(xs, i) = (reverse(Base.tail(reverse(axes(xs))))..., i)

batch(xs::AbstractArray{<:AbstractArray}) = stack(xs)

function batch(xs::Vector{<:Tuple})
    @assert length(xs) > 0 "Input should be non-empty"
    n = length(first(xs))
    @assert all(length.(xs) .== n) "Cannot batch tuples with different lengths"
    return ntuple(i -> batch([x[i] for x in xs]), n)
end

function batch(xs::Vector{<:NamedTuple})
    @assert length(xs) > 0 "Input should be non-empty"
    all_keys = [sort(collect(keys(x))) for x in xs]
    ks = all_keys[1]
    @assert all(==(ks), all_keys) "Cannot batch named tuples with different keys"
    return NamedTuple(k => batch([x[k] for x in xs]) for k in ks)
end

function batch(xs::Vector{<:Dict})
    @assert length(xs) > 0 "Input should be non-empty"
    all_keys = [sort(collect(keys(x))) for x in xs]
    ks = all_keys[1]
    @assert all(==(ks), all_keys) "cannot batch dicts with different keys"
    return Dict(k => batch([x[k] for x in xs]) for k in ks)
end

"""
    unbatch(x)

Reverse of the [`batch`](@ref) operation,
unstacking the last dimension of the array `x`.

See also [`unstack`](@ref) and [`chunk`](@ref).

# Examples

```jldoctest
julia> unbatch([1 3 5 7;
                2 4 6 8])
4-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
```                                                                                          
"""
unbatch(x::AbstractArray) = [getobs(x, i) for i in 1:numobs(x)]
unbatch(x::AbstractVector) = x

"""
    batchseq(seqs, val = 0)

Take a list of `N` sequences, and turn them into a single sequence where each
item is a batch of `N`. Short sequences will be padded by `val`.

# Examples

```jldoctest
julia> batchseq([[1, 2, 3], [4, 5]], 0)
3-element Vector{Vector{Int64}}:
 [1, 4]
 [2, 5]
 [3, 0]
```
"""
function batchseq(xs, val = 0)
    n = maximum(numobs, xs)
    xs_ = [rpad_constant(x, n, val; dims=ndims(x)) for x in xs]
    return [batch([getobs(xs_[j], i) for j = 1:length(xs_)]) for i = 1:n]
end

"""
    batch_sequence(seqs; pad = 0)

Take a list of `N` sequences `seqs`, 
where the `i`-th sequence is an array with last dimension `Li`,
and turn the into a single array with size `(..., Lmax, N)`.

The sequences need to have the same size, except for the last dimension.

Short sequences will be padded by `pad`.

See also [`batch`](@ref).

# Examples

```jldoctest
julia> batch_sequence([[1, 2, 3], [10, 20]])
3×2 Matrix{Int64}:
 1  10
 2  20
 3   0

julia> seqs = (ones(2, 3), fill(2.0, (2, 5)))
([1.0 1.0 1.0; 1.0 1.0 1.0], [2.0 2.0 … 2.0 2.0; 2.0 2.0 … 2.0 2.0])

julia> batch_sequence(xs, pad=-1)
2×5×2 Array{Float64, 3}:
[:, :, 1] =
 1.0  1.0  1.0  -1.0  -1.0
 1.0  1.0  1.0  -1.0  -1.0

[:, :, 2] =
 2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0
```
"""
function batch_sequence(xs, pad = 0)
    sz = size(xs[1])[1:end-1]
    @assert all(x -> size(x)[1:end-1] == sz, xs) "Array dimensions do not match."
    n = ndims(xs[1])
    Lmax = maximum(numobs, xs)
    padded_seqs = [rpad_constant(x, Lmax, pad, dims=n) for x in xs]
    return batch(padded_seqs)
end
