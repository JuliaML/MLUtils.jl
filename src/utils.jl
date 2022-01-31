# Some general utility functions. Many of them were part of Flux.jl

"""
    unsqueeze(xs, dim)

Return `xs` reshaped into an array one dimensionality higher than `xs`,
where `dim` indicates in which dimension `xs` is extended.
See also [`flatten`](@ref), [`stack`](@ref).

# Examples

```jldoctest
julia> unsqueeze([1 2; 3 4], 2)
2×1×2 Array{Int64, 3}:
[:, :, 1] =
 1
 3
[:, :, 2] =
 2
 4

julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> unsqueeze(xs, 1)
1×3 Matrix{Vector{Int64}}:
 [1, 2]  [3, 4]  [5, 6]
```
"""
function unsqueeze(xs::AbstractArray, dim::Integer)
    sz = ntuple(i -> i < dim ? size(xs, i) : i == dim ? 1 : size(xs, i - 1), ndims(xs) + 1)
    return reshape(xs, sz)
end

"""
    unsqueeze(dim)

Returns a function which, acting on an array, inserts a dimension of size 1 at `dim`.

# Examples

```jldoctest
julia> rand(21, 22, 23) |> unsqueeze(2) |> size
(21, 1, 22, 23)
```
"""
unsqueeze(dim::Integer) = Base.Fix2(unsqueeze, dim)

Base.show_function(io::IO, u::Base.Fix2{typeof(unsqueeze)}, ::Bool) = print(io, "unsqueeze(", u.x, ")")

"""
    stack(xs, dim)

Concatenate the given `Array` of `Array`s `xs` into a single `Array` along the
given dimension `dim`.

# Examples

```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> stack(xs, 1)
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> cat(xs, dims=1)
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
```
"""
stack(xs, dim) = cat(unsqueeze.(xs, dim)..., dims=dim)

"""
    unstack(xs, dim)

Unroll the given `xs` into an `Array` of `Array`s along the given dimension `dim`.

# Examples

```jldoctest
julia> unstack([1 3 5 7; 2 4 6 8], 2)
4-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
```
"""
unstack(xs, dim) = [copy(selectdim(xs, dim, i)) for i in 1:size(xs, dim)]

"""
    chunk(xs, n)

Split `xs` into `n` parts.

# Examples

```jldoctest
julia> chunk(1:10, 3)
3-element Vector{UnitRange{Int64}}:
 1:4
 5:8
 9:10

julia> chunk(collect(1:10), 3)
3-element Vector{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}:
 [1, 2, 3, 4]
 [5, 6, 7, 8]
 [9, 10]
```
"""
chunk(xs, n) = collect(Iterators.partition(xs, ceil(Int, length(xs)/n)))

"""
    frequencies(xs)

Count the number of times that each element of `xs` appears.

# Examples

```jldoctest
julia> frequencies(['a','b','b'])
Dict{Char, Int64} with 2 entries:
  'a' => 1
  'b' => 2
```
"""
function frequencies(xs)
    fs = Dict{eltype(xs),Int}()
    for x in xs
        fs[x] = get(fs, x, 0) + 1
    end
    return fs
end

"""
    batch(xs)

Batch the arrays in `xs` into a single array with 
an extra dimension.

If the elements of `xs` are tuples, named tuples, or dicts, 
the output will be of the same type. 

See also [`unbatch`](@ref).

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

function batch(xs::AbstractVector{<:AbstractArray{T,N}}) where {T, N}
    return stack(xs, N+1)
end

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
    NamedTuple(k => batch([x[k] for x in xs]) for k in ks)
end

function batch(xs::Vector{<:Dict})
    @assert length(xs) > 0 "Input should be non-empty"
    all_keys = [sort(collect(keys(x))) for x in xs]
    ks = all_keys[1]
    @assert all(==(ks), all_keys) "cannot batch dicts with different keys"
    Dict(k => batch([x[k] for x in xs]) for k in ks)
end

"""
    unbatch(x)

Reverse of the [`batch`](@ref) operation,
unstacking the last dimension of the array `x`.
See also [`unstack`](@ref).

# Examples

```jldoctest
julia> unbatch([1 3 5 7;
                     2 4 6 8])
4-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
"""
unbatch(x::AbstractArray) = unstack(x, ndims(x))
unbatch(x::AbstractVector) = x

"""
    rpad(v::AbstractVector, n::Integer, p)

Return the given sequence padded with `p` up to a maximum length of `n`.

# Examples

```jldoctest
julia> rpad([1, 2], 4, 0)
4-element Vector{Int64}:
 1
 2
 0
 0

julia> rpad([1, 2, 3], 2, 0)
3-element Vector{Int64}:
 1
 2
 3
```
"""
Base.rpad(v::AbstractVector, n::Integer, p) = [v; fill(p, max(n - length(v), 0))]
# TODO Piracy


"""
    batchseq(seqs, pad)

Take a list of `N` sequences, and turn them into a single sequence where each
item is a batch of `N`. Short sequences will be padded by `pad`.

# Examples

```jldoctest
julia> batchseq([[1, 2, 3], [4, 5]], 0)
3-element Vector{Vector{Int64}}:
 [1, 4]
 [2, 5]
 [3, 0]
```
"""
function batchseq(xs, pad = nothing, n = maximum(length(x) for x in xs))
    xs_ = [rpad(x, n, pad) for x in xs]
    [batch([xs_[j][i] for j = 1:length(xs_)]) for i = 1:n]
end

"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output,
preserving the size of the last dimension.
See also [`unsqueeze`](@ref).

# Examples

```jldoctest
julia> rand(3,4,5) |> flatten |> size
(12, 5)
```
"""
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

"""
    normalise(x; dims=ndims(x), ϵ=1e-5)

Normalise `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
Per default, `dims` is the last dimension. 
`ϵ` is a small additive factor added to the denominator for numerical stability.
"""
function normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
    μ = mean(x, dims=dims)
    #   σ = std(x, dims=dims, mean=μ, corrected=false) # use this when Zygote#478 gets merged
    σ = std(x, dims=dims, corrected=false)
    return (x .- μ) ./ (σ .+ ϵ)
end

ofeltype(x, y) = convert(float(eltype(x)), y)
epseltype(x) = eps(float(eltype(x)))