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
    chunk(x, n; [dims])

Split `x` into `n` parts. The parts contain the same number of elements
except possibly for the last one that can be smaller.

If `x` is an array, `dims` can be used to specify along which dimension to 
split (defaults to the last dimension).

# Examples

```jldoctest
julia> chunk(1:10, 3)
3-element Vector{UnitRange{Int64}}:
 1:4
 5:8
 9:10

julia> x = reshape(collect(1:20), (5, 4))
5×4 Matrix{Int64}:
 1   6  11  16
 2   7  12  17
 3   8  13  18
 4   9  14  19
 5  10  15  20

julia> xs = chunk(x, 2, dims=1)
2-element Vector{SubArray{Int64, 2, Matrix{Int64}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}}:
 [1 6 11 16; 2 7 12 17; 3 8 13 18]
 [4 9 14 19; 5 10 15 20]

julia> xs[1]
3×4 view(::Matrix{Int64}, 1:3, :) with eltype Int64:
 1  6  11  16
 2  7  12  17
 3  8  13  18
```
"""
chunk(x; size::Int) = collect(Iterators.partition(x, size))
chunk(x, n::Int) = chunk(x; size = ceil(Int, length(x) / n))

function chunk(x::AbstractArray; size::Int, dims::Int=ndims(x))
    idxs = _partition_idxs(x, size, dims)
    [selectdim(x, dims, i) for i in idxs]
end
chunk(x::AbstractArray, n::Int; dims::Int=ndims(x)) = chunk(x; size = ceil(Int, size(x, dims) / n), dims)

function rrule(::typeof(chunk), x::AbstractArray; size::Int, dims::Int=ndims(x))
    # this is the implementation of chunk
    idxs = _partition_idxs(x, size, dims)
    y = [selectdim(x, dims, i) for i in idxs]
    valdims = Val(dims)
    chunk_pullback(dy) = (NoTangent(), ∇chunk(unthunk(dy), x, idxs, valdims))

    return y, chunk_pullback
end

_partition_idxs(x, size, dims) = Iterators.partition(axes(x, dims), size)

# Similar to ∇eachslice  https://github.com/JuliaDiff/ChainRules.jl/blob/8108a77a96af5d4b0c460aac393e44f8943f3c5e/src/rulesets/Base/indexing.jl#L77
function ∇chunk(dys, x::AbstractArray, idxs, vd::Val{dim}) where {dim}
    i1 = findfirst(dy -> !(dy isa AbstractZero), dys)
    if i1 === nothing  # all slices are Zero!
        return _zero_fill!(similar(x, float(eltype(x))))
    end
    T = promote_type(eltype(dys[i1]), eltype(x))
    # The whole point of this gradient is that we can allocate one `dx` array:
    dx = similar(x, T)
    for (k, i) in enumerate(idxs)
        slice = selectdim(dx, dim, i)
        if dys[k] isa AbstractZero
            _zero_fill!(slice)  # Avoids this: copyto!([1,2,3], ZeroTangent()) == [0,2,3]
        else
            copyto!(slice, dys[k])
        end
    end
    return ProjectTo(x)(dx)
end

_zero_fill!(dx::AbstractArray{<:Number}) = fill!(dx, zero(eltype(dx)))
_zero_fill!(dx::AbstractArray) = map!(zero, dx, dx)

function rrule(::typeof(∇chunk), dys, x, idxs, vd::Val{dim}) where dim
    n = length(dys)
    function ∇∇chunk(dz_raw)
        dz = chunk(unthunk(dz_raw), n; dims=dim)
        return (NoTangent(), dz, NoTangent(), NoTangent(), NoTangent())
    end
    return ∇chunk(dys, x, idxs, vd), ∇∇chunk
end

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

```jldoctest
julia> x = [:yes, :no, :maybe, :yes];

julia> group_indices(x)
Dict{Symbol, Vector{Int64}} with 3 entries:
  :yes   => [1, 4]
  :maybe => [3]
  :no    => [2]
```
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

function batch(xs::AbstractArray{<:AbstractArray})
    # Don't use stack(xs, dims=N+1), it is much slower.
    # Here we do reduce(vcat, xs) along with some reshapes.
    szxs = size(xs)
    @assert length(xs) > 0 "Minimum batch size is 1."
    szx = size(xs[1])
    @assert all(x -> size(x) == szx, xs) "All arrays must be of the same size."
    vxs = vec(vec.(xs))
    y = reduce(vcat, vxs)
    return reshape(y, szx..., szxs...)
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
```                                                                                          
"""
unbatch(x::AbstractArray) = [getobs(x, i) for i in 1:numobs(x)]
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

"""
    ones_like(x, [element_type=eltype(x)], [dims=size(x)]))

Create an array with the given element type and size, based upon the given source array `x`.
All element of the new array will be set to 1. 
The second and third arguments are both optional, defaulting to the given array's eltype and
size. The dimensions may be specified as an integer or as a tuple argument.

See also [`zeros_like`](@ref) and [`fill_like`](@ref).

# Examples

```julia-repl
julia> x = rand(Float32, 2)
2-element Vector{Float32}:
 0.8621633
 0.5158395

julia> ones_like(x, (3, 3))
3×3 Matrix{Float32}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> using CUDA

julia> x = CUDA.rand(2, 2)
2×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 0.82297   0.656143
 0.701828  0.391335

julia> ones_like(x, Float64)
2×2 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 1.0  1.0
 1.0  1.0
```
"""
ones_like(x::AbstractArray, T::Type, sz=size(x)) = fill!(similar(x, T, sz), 1)
ones_like(x::AbstractArray, sz=size(x)) = ones_like(x, eltype(x), sz)


"""
    zeros_like(x, [element_type=eltype(x)], [dims=size(x)]))


Create an array with the given element type and size, based upon the given source array `x`.
All element of the new array will be set to 0. 
The second and third arguments are both optional, defaulting to the given array's eltype and
size. The dimensions may be specified as an integer or as a tuple argument.

See also [`ones_like`](@ref) and [`fill_like`](@ref).

# Examples

```julia-repl
julia> x = rand(Float32, 2)
2-element Vector{Float32}:
 0.4005432
 0.36934233

julia> zeros_like(x, (3, 3))
3×3 Matrix{Float32}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> using CUDA

julia> x = CUDA.rand(2, 2)
2×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 0.0695155  0.667979
 0.558468   0.59903

julia> zeros_like(x, Float64)
2×2 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 0.0  0.0
 0.0  0.0
```
"""
zeros_like(x::AbstractArray, T::Type, sz=size(x)) = fill!(similar(x, T, sz), 0)
zeros_like(x::AbstractArray, sz=size(x)) = zeros_like(x, eltype(x), sz)

"""
    rand_like([rng=default_rng()], x, [element_type=eltype(x)], [dims=size(x)])

Create an array with the given element type and size, based upon the given source array `x`.
All element of the new array will be set to a random value.
The last two arguments are both optional, defaulting to the given array's eltype and
size. The dimensions may be specified as an integer or as a tuple argument.

The default random number generator is used, unless a custom one is passed in explicitly
as the first argument.

See also `Base.rand` and [`randn_like`](@ref).

# Examples

```julia-repl
julia> x = ones(Float32, 2)
2-element Vector{Float32}:
 1.0
 1.0

julia> rand_like(x, (3, 3))
3×3 Matrix{Float32}:
 0.780032  0.920552  0.53689
 0.121451  0.741334  0.5449
 0.55348   0.138136  0.556404

julia> using CUDA

julia> CUDA.ones(2, 2)
2×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 1.0  1.0
 1.0  1.0

julia> rand_like(x, Float64)
2×2 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 0.429274  0.135379
 0.718895  0.0098756
```
"""
rand_like(x::AbstractArray, T::Type, sz=size(x)) = rand!(similar(x, T, sz))
rand_like(x::AbstractArray, sz=size(x)) = rand_like(x, eltype(x), sz)
rand_like(rng::AbstractRNG, x::AbstractArray, T::Type, sz=size(x)) = rand!(rng, similar(x, T, sz))
rand_like(rng::AbstractRNG, x::AbstractArray, sz=size(x)) = rand_like(rng, x, eltype(x), sz)

"""
    randn_like([rng=default_rng()], x, [element_type=eltype(x)], [dims=size(x)])

Create an array with the given element type and size, based upon the given source array `x`.
All element of the new array will be set to a random value drawn from a normal distribution.
The last two arguments are both optional, defaulting to the given array's eltype and
size. The dimensions may be specified as an integer or as a tuple argument.

The default random number generator is used, unless a custom one is passed in explicitly
as the first argument.

See also `Base.randn` and [`rand_like`](@ref).

# Examples
```julia-repl
julia> x = ones(Float32, 2)
2-element Vector{Float32}:
 1.0
 1.0

julia> randn_like(x, (3, 3))
3×3 Matrix{Float32}:
 -0.385331    0.956231   0.0745102
  1.43756    -0.967328   2.06311
  0.0482372   1.78728   -0.902547

julia> using CUDA

julia> CUDA.ones(2, 2)
2×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 1.0  1.0
 1.0  1.0

julia> randn_like(x, Float64)
2×2 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 -0.578527   0.823445
 -1.01338   -0.612053
```
"""
randn_like(x::AbstractArray, T::Type, sz=size(x)) = randn!(similar(x, T, sz))
randn_like(x::AbstractArray, sz=size(x)) = randn_like(x, eltype(x), sz)
randn_like(rng::AbstractRNG, x::AbstractArray, T::Type, sz=size(x)) = randn!(rng, similar(x, T, sz))
randn_like(rng::AbstractRNG, x::AbstractArray, sz=size(x)) = randn_like(rng, x, eltype(x), sz)

"""
    fill_like(x, val, [element_type=eltype(x)], [dims=size(x)]))

Create an array with the given element type and size, based upon the given source array `x`.
All element of the new array will be set to `val`. 
The third and fourth arguments are both optional, defaulting to the given array's eltype and
size. The dimensions may be specified as an integer or as a tuple argument.

See also [`zeros_like`](@ref) and [`ones_like`](@ref).

# Examples

```julia-repl
julia> x = rand(Float32, 2)
2-element Vector{Float32}:
 0.16087806
 0.89916044

julia> fill_like(x, 1.7, (3, 3))
3×3 Matrix{Float32}:
 1.7  1.7  1.7
 1.7  1.7  1.7
 1.7  1.7  1.7

julia> using CUDA

julia> x = CUDA.rand(2, 2)
2×2 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 0.803167  0.476101
 0.303041  0.317581

julia> fill_like(x, 1.7, Float64)
2×2 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 1.7  1.7
 1.7  1.7
```
"""
fill_like(x::AbstractArray, val, T::Type, sz=size(x)) = fill!(similar(x, T, sz), val)
fill_like(x::AbstractArray, val, sz=size(x)) = fill_like(x, val, eltype(x), sz)

@non_differentiable zeros_like(::Any...)
@non_differentiable ones_like(::Any...)
@non_differentiable rand_like(::Any...)
@non_differentiable randn_like(::Any...)

function rrule(::typeof(fill_like), x::AbstractArray, val, T::Type, sz)
    function fill_like_pullback(Δ)
        return (NoTangent(), ZeroTangent(), sum(Δ), NoTangent(), NoTangent())
    end
    return fill_like(x, val, T, sz), fill_like_pullback
end
