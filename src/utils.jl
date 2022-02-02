# Some general utility functions. Many of them were part of Flux.jl

"""
    unsqueeze(x; dims)

Return `x` reshaped into an array one dimensionality higher than `x`,
where `dims` indicates in which dimension `x` is extended.

See also [`flatten`](@ref), [`stack`](@ref).

# Examples

```jldoctest
julia> unsqueeze([1 2; 3 4], dims=2)
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

julia> unsqueeze(xs, dims=1)
1×3 Matrix{Vector{Int64}}:
 [1, 2]  [3, 4]  [5, 6]
```
"""
function unsqueeze(x::AbstractArray; dims::Int)
    sz = ntuple(i -> i < dims ? size(x, i) : i == dims ? 1 : size(x, i - 1), ndims(x) + 1)
    return reshape(x, sz)
end

"""
    unsqueeze(; dims)

Returns a function which, acting on an array, inserts a dimension of size 1 at `dims`.

# Examples

```jldoctest
julia> rand(21, 22, 23) |> unsqueeze(dims=2) |> size
(21, 1, 22, 23)
```
"""
unsqueeze(; dims::Int) = Base.Fix2(_unsqueeze, dims)
_unsqueeze(x, dims) = unsqueeze(x; dims)

Base.show_function(io::IO, u::Base.Fix2{typeof(_unsqueeze)}, ::Bool) = print(io, "unsqueeze(dims=", u.x, ")")

"""
    stack(xs; dims)

Concatenate the given array of arrays `xs` into a single array along the
given dimension `dims`.

See also [`stack`](@ref) and [`batch`](@ref).

# Examples

```jldoctest
julia> xs = [[1, 2], [3, 4], [5, 6]]
3-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]

julia> stack(xs, dims=1)
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> stack(xs, dims=2)
2×3 Matrix{Int64}:
 1  3  5
 2  4  6

julia> stack(xs, dims=3)
2×1×3 Array{Int64, 3}:
[:, :, 1] =
 1
 2

[:, :, 2] =
 3
 4

[:, :, 3] =
 5
 6
```
"""
stack(xs; dims::Int) = cat(unsqueeze.(xs; dims)...; dims)

"""
    unstack(xs; dims)

Unroll the given `xs` into an array of arrays along the given dimension `dims`.

See also [`stack`](@ref) and [`unbatch`](@ref).

# Examples

```jldoctest
julia> unstack([1 3 5 7; 2 4 6 8], dims=2)
4-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
```
"""
unstack(xs; dims::Int) = [copy(selectdim(xs, dims, i)) for i in 1:size(xs, dims)]

"""
    chunk(x, n)

Split `x` into `n` parts.

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
    group_counts(x)

Count the number of times that each element of `x` appears.

See also [`group_indices`](@ref)
# Examples

```jldoctest
julia> group_counts(['a', 'b', 'b'])
Dict{Char, Int64} with 2 entries:
  'a' => 1
  'b' => 2
```
"""
function group_counts(x)
    fs = Dict{eltype(x),Int}()
    for a in x
        fs[a] = get(fs, a, 0) + 1
    end
    return fs
end

"""
    group_indices(x) -> Dict

Computes the indices of elements in the vector `x` for each distinct value contained. 
This information is useful for resampling strategies, such as stratified sampling.

See also [`group_counts`](@ref).

# Examples

```julia
julia> x = [:yes, :no, :maybe, :yes];

julia> group_indices(x)
Dict{Symbol, Vector{Int64}} with 3 entries:
  :yes   => [1, 4]
  :maybe => [3]
  :no    => [2]

"""
function group_indices(classes::T) where T<:AbstractVector
    dict = Dict{eltype(T), Vector{Int}}()
    for (idx, elem) in enumerate(classes)
        if !haskey(dict, elem)
            push!(dict, elem => [idx])
        else
            push!(dict[elem], idx)
        end
    end
    return dict
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
    return stack(xs, dims=N+1)
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
unbatch(x::AbstractArray) = unstack(x, dims=ndims(x))
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

Normalise the array `x` to mean 0 and standard deviation 1 across the dimension(s) given by `dims`.
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