# Some general utility functions. Many of them were part of Flux.jl

"""
    unsqueeze(x; dims)

Return `x` reshaped into an array one dimensionality higher than `x`,
where `dims` indicates in which dimension `x` is extended.
`dims` can be an integer between 1 and `ndims(x)+1`.

See also [`flatten`](@ref), `stack`.

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
function unsqueeze(x::AbstractArray{T,N}; dims::Int) where {T, N}
    @assert 1 <= dims <= N + 1
    sz = ntuple(i -> i < dims ? size(x, i) : i == dims ? 1 : size(x, i - 1), Val(N + 1))
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
    unstack(xs; dims)

Unroll the given `xs` into an array of arrays along the given dimension `dims`.

It is the inverse operation of [stack](https://docs.julialang.org/en/v1/base/arrays/#Base.stack).

See also [`unbatch`](@ref) and [`chunk`](@ref).

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
unstack(xs; dims) = _unstack(dims, xs)

dims2unstack(::Val{dims}) where dims = dims2unstack(dims)
dims2unstack(dims::Integer) = dims

@inline _unstack(dims, xs) = [copy(selectdim(xs, dims2unstack(dims), i)) for i in axes(xs, dims2unstack(dims))]

"""
    chunk(x, n; [dims])
    chunk(x; [size, dims])

Split `x` into `n` parts or alternatively, if `size` is an integer, into equal chunks of size `size`. 
The parts contain the same number of elements except possibly for the last one that can be smaller.

In case `size` is a collection of integers instead, the elements of `x` are split into chunks of
the given sizes.

If `x` is an array, `dims` can be used to specify along which dimension to
split (defaults to the last dimension). `dims` can be an integer or a `Val`;
pass `dims = Val(d)` to make the return type inferable when `d` is not a
compile-time constant.

# Examples

```jldoctest
julia> chunk(1:10, 3)
3-element Vector{UnitRange{Int64}}:
 1:4
 5:8
 9:10

julia> chunk(1:10; size = 2)
5-element Vector{UnitRange{Int64}}:
 1:2
 3:4
 5:6
 7:8
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

julia> xes = chunk(x; size = 2, dims = 2)
2-element Vector{SubArray{Int64, 2, Matrix{Int64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}}:
 [1 6; 2 7; … ; 4 9; 5 10]
 [11 16; 12 17; … ; 14 19; 15 20]

julia> xes[2]
5×2 view(::Matrix{Int64}, :, 3:4) with eltype Int64:
 11  16
 12  17
 13  18
 14  19
 15  20

julia> chunk(1:6; size = [2, 4])
2-element Vector{UnitRange{Int64}}:
 1:2
 3:6
```
"""
chunk(x; size::Int) = collect(Iterators.partition(x, size))

chunk(x, n::Int) = chunk(x; size = cld(length(x), n))

Base.@constprop :aggressive chunk(x::AbstractArray, n::Int; dims::Union{Int,Val}=ndims(x)) =
    chunk(x; size = cld(size(x, _unval(dims)), n), dims)

# `@constprop` and the `Val` barrier in `_chunk` let a constant `dims` infer the view type
Base.@constprop :aggressive function chunk(x::AbstractArray; size, dims::Union{Int,Val}=ndims(x))
    idxs = _partition_idxs(x, size, _unval(dims))
    return _chunk(x, idxs, Val(_unval(dims)))
end

_chunk(x::AbstractArray, idxs, vd::Val) = [_selectdim(x, vd, i) for i in idxs]

_unval(d::Int) = d
_unval(::Val{d}) where {d} = d


"""
    chunk(x, partition_idxs; [npartitions, dims])

Partition the array `x` along the dimension `dims` according to the indexes 
in `partition_idxs`.

`partition_idxs` must be sorted and contain only positive integers 
between 1 and the number of partitions. 

If the number of partition `npartitions` is not provided, 
it is inferred from `partition_idxs`.

If `dims` is not provided, it defaults to the last dimension.

See also [`unbatch`](@ref).

# Examples

```jldoctest
julia> x = reshape([1:10;], 2, 5)
2×5 Matrix{Int64}:
 1  3  5  7   9
 2  4  6  8  10

julia> chunk(x, [1, 2, 2, 3, 3])
3-element Vector{SubArray{Int64, 2, Matrix{Int64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}}:
 [1; 2;;]
 [3 5; 4 6]
 [7 9; 8 10]
```
"""
function chunk(x::AbstractArray{T,N}, partition_idxs::AbstractVector; 
        npartitions=nothing, dims=ndims(x)) where {T, N}
    @assert issorted(partition_idxs) "partition_idxs must be sorted"
    m = npartitions === nothing ? maximum(partition_idxs) : npartitions
    degrees = NNlib.scatter(+, ones_like(partition_idxs), partition_idxs, dstsize=(m,))
    return chunk(x; size=degrees, dims)
end

# work around https://github.com/JuliaML/MLUtils.jl/issues/103
_selectdim(x::AbstractArray, dims::Int, i) = selectdim(x, dims, i)
_selectdim(x::AbstractArray, dims::Int, i::UnitRange) = _selectdim(x, Val(dims), i)
_selectdim(x::AbstractArray, ::Val{dims}, i) where {dims} = selectdim(x, dims, i)

function _selectdim(x::AbstractArray{T,N}, ::Val{dims}, i::UnitRange) where {T,N,dims}
    return view(x, ntuple(_ -> Colon(), dims-1)..., i, ntuple(_ -> Colon(), N-dims)...)
end

function rrule(::typeof(chunk), x::AbstractArray; size, dims::Union{Int,Val}=ndims(x))
    # This is the implementation of chunk
    idxs = _partition_idxs(x, size, _unval(dims))
    valdims = Val(_unval(dims))
    y = _chunk(x, idxs, valdims)
    # Capture lightweight stand-ins for `x` instead of `x` itself, so the input
    # data isn't kept alive by the pullback: `ProjectTo(x)` carries `x`'s axes
    # and restores its structure, and `similar(x, 0)` carries `x`'s array/element
    # type (for allocating the gradient) without holding any data.
    project = ProjectTo(x)
    xproto = similar(x, 0)
    chunk_pullback(dy) = (NoTangent(), ∇chunk(unthunk(dy), project, xproto, idxs, valdims))

    return y, chunk_pullback
end

_partition_idxs(x, size::Int, dims::Int) = Iterators.partition(axes(x, dims), size)

_partition_idxs(x, size, dims::Int) = _partition_idxs(x, collect(size), dims)

function _partition_idxs(x, size::AbstractVector{<:Integer}, dims::Int)
    n = length(axes(x, dims))
    cumsz = cumsum(size)
    if cumsz[end] != n
        throw(ArgumentError("The sum of the sizes must be equal to $n, the length of the dimension."))
    end
    return [(i==1 ? 1 : cumsz[i-1]+1):cumsz[i]  for i=1:length(cumsz)]
end

@non_differentiable _partition_idxs(::Any...)

# Similar to ∇eachslice  https://github.com/JuliaDiff/ChainRules.jl/blob/8108a77a96af5d4b0c460aac393e44f8943f3c5e/src/rulesets/Base/indexing.jl#L77
# `project` is `ProjectTo(x)` and `xproto` is `similar(x, 0)`, both derived from
# the original input `x`. Together they supply `x`'s axes, array type and element
# type, so we never need to capture `x` (and its data) itself.
function ∇chunk(dys_raw, project::ProjectTo, xproto, idxs, vd::Val{dim}) where {dim}
    dys = unthunk.(unthunk(dys_raw))  # https://github.com/FluxML/Zygote.jl/pull/966#issuecomment-2569227272
    i1 = findfirst(dy -> !(dy isa AbstractZero), dys)
    if i1 === nothing  # all slices are Zero!
        return project(_zero_fill!(similar(xproto, float(eltype(xproto)), project.axes)))
    end
    T = promote_type(eltype(dys[i1]), eltype(xproto))
    # The whole point of this gradient is that we can allocate one `dx` array.
    # `xproto` gives it `x`'s array type and `project.axes` its shape, without
    # needing `x` itself; `project` then restores `x`'s structure.
    dx = similar(xproto, T, project.axes)
    for (k, i) in enumerate(idxs)
        slice = _selectdim(dx, dim, i)
        if dys[k] isa AbstractZero
            _zero_fill!(slice)  # Avoids this: copyto!([1,2,3], ZeroTangent()) == [0,2,3]
        else
            copyto!(slice, dys[k])
        end
    end
    return project(dx)
end

_zero_fill!(dx::AbstractArray{<:Number}) = fill!(dx, zero(eltype(dx)))
_zero_fill!(dx::AbstractArray) = map!(zero, dx, dx)

function rrule(::typeof(∇chunk), dys, project, xproto, idxs, vd::Val{dim}) where dim
    n = length(dys)
    function ∇∇chunk(dz_raw)
        dz = chunk(unthunk(dz_raw), n; dims=vd)
        return (NoTangent(), dz, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return ∇chunk(dys, project, xproto, idxs, vd), ∇∇chunk
end

"""
    group_counts(x) -> Dict

Count the number of times that each element of `x` appears.

See also [`group_indices`](@ref)
# Examples

```jldoctest
julia> counts = group_counts(['a', 'b', 'b']);

julia> sort(collect(counts))  # a Dict has no defined order
2-element Vector{Pair{Char, Int64}}:
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

julia> indices = group_indices(x);

julia> sort(collect(indices))  # a Dict has no defined order
3-element Vector{Pair{Symbol, Vector{Int64}}}:
 :maybe => [3]
    :no => [2]
   :yes => [1, 4]
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
    rpad_constant(v::AbstractArray, n::Union{Integer, Tuple}, val = 0; dims=:)

Return the given sequence padded with `val` along the dimensions `dims`
up to a maximum length in each direction specified by `n`.

# Examples
```jldoctest
julia> rpad_constant([1, 2], 4, -1) # passing with -1 up to size 4
4-element Vector{Int64}:
  1
  2
 -1
 -1

julia> rpad_constant([1, 2, 3], 2) # no padding if length is already greater than n
3-element Vector{Int64}:
 1
 2
 3

julia> rpad_constant([1 2; 3 4], 4; dims=1) # padding along the first dimension
4×2 Matrix{Int64}:
 1  2
 3  4
 0  0
 0  0

julia> rpad_constant([1 2; 3 4], 4) # padding along all dimensions by default
4×4 Matrix{Int64}:
 1  2  0  0
 3  4  0  0
 0  0  0  0
 0  0  0  0
```
"""
function rpad_constant(x::AbstractArray, n::Union{Integer, Tuple}, val=0; dims=:)
    ns = _rpad_pads(x, n, dims)
    return NNlib.pad_constant(x, ns, val; dims)
end

function _rpad_pads(x, n, dims)
    _dims = dims === Colon() ? (1:ndims(x)) : dims
    _n = n isa Integer ? ntuple(i -> n, length(_dims)) : n
    @assert length(_dims) == length(_n)
    ns = ntuple(i -> isodd(i) ? 0 : max(_n[i÷2] - size(x, _dims[i÷2]), 0), 2*length(_n))
    return ns
end

@non_differentiable _rpad_pads(::Any...)

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

This operation is also known as *standardization* or *z-score normalization*.
See also [`rescale`](@ref) for min-max scaling to the `[0, 1]` range.
"""
function normalise(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
    μ = mean(x, dims=dims)
    #   σ = std(x, dims=dims, mean=μ, corrected=false) # use this when Zygote#478 gets merged
    σ = std(x, dims=dims, corrected=false)
    return (x .- μ) ./ (σ .+ ϵ)
end

"""
    rescale(x; dims=ndims(x), ϵ=1e-5)

Rescale the array `x` to the `[0, 1]` range across the dimension(s) given by `dims`,
by subtracting the minimum and dividing by the range (maximum minus minimum).
Per default, `dims` is the last dimension.

This operation is also known as *min-max scaling* or *min-max normalization*.

`ϵ` is a small additive factor added to the denominator for numerical stability.
See also [`normalise`](@ref) for standardization to mean 0 and standard deviation 1.
"""
function rescale(x::AbstractArray; dims=ndims(x), ϵ=ofeltype(x, 1e-5))
    lo = minimum(x, dims=dims)
    hi = maximum(x, dims=dims)
    return (x .- lo) ./ (hi .- lo .+ ϵ)
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

"""
    trues_like(x, [dims=size(x)])

Equivalent to `fill_like(x, true, Bool, dims)`.

See also [`fill_like`] and [`falses_like`](@ref).
"""
trues_like(x::AbstractArray, sz=size(x)) = fill_like(x, true, Bool, sz)

"""
    falses_like(x, [dims=size(x)])

Equivalent to `fill_like(x, false, Bool, dims)`.

See also [`fill_like`] and [`trues_like`](@ref).
"""
falses_like(x::AbstractArray, sz=size(x)) = fill_like(x, false, Bool, sz)


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

"""
    batched_searchsortedfirst(xp, x; dims=1, rev=false)

Batched, GPU-friendly version of `Base.searchsortedfirst`.

For each query in `x`, return the index of the first element of the sorted
collection `xp` that is greater than or equal to it (the leftmost position at
which the query could be inserted while keeping `xp` sorted). Pass `rev=true`
when `xp` is sorted in descending order.

The search runs along dimension `dims` of `xp`. `x` and `xp` must have the same
size along all other dimensions (or `xp` may be a vector, in which case it is
searched for every query in `x`). The result has the same size as `x`, so that
for `dims=1`, `xp[result[i, j], j] >= x[i, j]`.

See also [`batched_searchsortedlast`](@ref).

# Examples

```jldoctest
julia> v = [1, 3, 5, 7, 9];

julia> batched_searchsortedfirst(v, [4, 5, 8])
3-element Vector{Int64}:
 3
 3
 5

julia> xp = [1 2; 3 4; 5 6];   # two sorted columns: [1,3,5] and [2,4,6]

julia> batched_searchsortedfirst(xp, [2 5; 4 1])
2×2 Matrix{Int64}:
 2  3
 3  1
```
"""
function batched_searchsortedfirst(xp::AbstractArray, x::AbstractArray; dims::Integer=1, rev::Bool=false)
    cmp = _searchsorted_count(xp, x, dims, rev ? (>) : (<))
    return cmp .+ 1
end

"""
    batched_searchsortedlast(xp, x; dims=1, rev=false)

Batched, GPU-friendly version of `Base.searchsortedlast`.

For each query in `x`, return the index of the last element of the sorted
collection `xp` that is less than or equal to it (the rightmost position at
which the query could be inserted while keeping `xp` sorted). Pass `rev=true`
when `xp` is sorted in descending order.

The search runs along dimension `dims` of `xp`. `x` and `xp` must have the same
size along all other dimensions (or `xp` may be a vector, in which case it is
searched for every query in `x`). The result has the same size as `x`, so that
for `dims=1`, `xp[result[i, j], j] <= x[i, j]`.

See also [`batched_searchsortedfirst`](@ref).

# Examples

```jldoctest
julia> v = [1, 3, 5, 7, 9];

julia> batched_searchsortedlast(v, [4, 5, 8])
3-element Vector{Int64}:
 2
 3
 4

julia> xp = [1 2; 3 4; 5 6];   # two sorted columns: [1,3,5] and [2,4,6]

julia> batched_searchsortedlast(xp, [2 5; 4 1])
2×2 Matrix{Int64}:
 1  2
 2  0
```
"""
function batched_searchsortedlast(xp::AbstractArray, x::AbstractArray; dims::Integer=1, rev::Bool=false)
    return _searchsorted_count(xp, x, dims, rev ? (>=) : (<=))
end

# Count, for each query in `x`, how many elements of `xp` satisfy `cmp(xp, x)`
# along dimension `dims`. The comparison is broadcast over an extra axis, so the
# whole computation is vectorized and stays on the GPU. `xp` may be a vector, in
# which case it is broadcast against every batch of `x`.
function _searchsorted_count(xp::AbstractArray, x::AbstractArray, dims::Integer, cmp)
    xpe = unsqueeze(xp; dims = dims + 1)   # search dim stays at `dims`
    xe = unsqueeze(x; dims = dims)         # query dim moves to `dims + 1`
    counts = sum(cmp.(xpe, xe); dims)
    return dropdims(counts; dims)
end

@non_differentiable batched_searchsortedfirst(::Any...)
@non_differentiable batched_searchsortedlast(::Any...)

"""
    topk(x, k; dims=1, rev=true) -> (values, indices)

Return the `k` largest elements of array `x` along dimension `dims`, and their
indices along that dimension. The result is sorted, with `values[1, ...]` the
largest element. Pass `rev=false` to return the `k` smallest elements instead.

`values` and `indices` are arrays equal in size to `x` except along `dims`, where
they have size `k`. The returned `indices` are the positions of the selected
elements along `dims`, so that for `dims=1`, `x[indices[i, j], j] == values[i, j]`.

This is a differentiable, multidimensional and GPU-friendly generalization of
`Base.partialsort`.

# Examples

```jldoctest
julia> x = [1 8 3; 6 2 5; 4 9 7];

julia> vals, inds = topk(x, 2; dims=2);

julia> vals
3×2 Matrix{Int64}:
 8  3
 6  5
 9  7

julia> inds
3×2 Matrix{Int64}:
 2  3
 1  3
 2  3

julia> topk([5, 1, 4, 2, 3], 3)
([5, 4, 3], [1, 3, 5])
```
"""
function topk(x::AbstractArray, k::Integer; dims::Integer=1, rev::Bool=true)
    n = size(x, dims)
    1 <= k <= n || throw(ArgumentError("k=$k must be in the range 1:size(x, dims)=$n"))
    perm = _sortperm(x, dims, rev)
    lin = _first_along(perm, k, dims)   # linear indices of the top-k slices
    vals = x[lin]
    inds = _idx_along_dim(lin, size(x), dims)
    return vals, inds
end

# `sortperm` rejects the `dims` keyword for vectors, where it is redundant anyway.
_sortperm(x::AbstractVector, dims::Integer, rev::Bool) = sortperm(x; rev)
_sortperm(x::AbstractArray, dims::Integer, rev::Bool) = sortperm(x; dims, rev)

# The first `k` slices along `dims`. Unlike `selectdim`, inferable for non-constant `dims`.
_first_along(x, k, dims) = x[ntuple(d -> d == dims ? (1:k) : (1:size(x, d)), ndims(x))...]

# Convert linear indices `lin` into x to positions along dimension `dims`.
# Fully broadcasted so it stays on the GPU.
function _idx_along_dim(lin, sz::Dims, dims::Integer)
    stride = prod(ntuple(d -> d < dims ? sz[d] : 1, length(sz)))   # 1 when dims == 1
    dimsize = sz[dims]
    return @. mod(div(lin - 1, stride), dimsize) + 1
end

@non_differentiable _idx_along_dim(::Any...)

function rrule(::typeof(topk), x::AbstractArray, k::Integer; dims::Integer=1, rev::Bool=true)
    n = size(x, dims)
    1 <= k <= n || throw(ArgumentError("k=$k must be in the range 1:size(x, dims)=$n"))
    perm = _sortperm(x, dims, rev)
    lin = _first_along(perm, k, dims)
    vals = x[lin]
    inds = _idx_along_dim(lin, size(x), dims)
    function topk_pullback(Δ)
        Δvals = Δ[1]                          # only `values` carries a gradient
        if Δvals === nothing || Δvals isa AbstractZero
            return (NoTangent(), ZeroTangent(), NoTangent())
        end
        Δvals = unthunk(Δvals)
        x̄ = zero(x)
        x̄[lin] = Δvals                       # scatter; the top-k indices are unique
        return (NoTangent(), x̄, NoTangent())
    end
    return (vals, inds), topk_pullback
end

"""
    sample_logits([rng], logits; dims=1, n=1, replace=true, k=nothing, p=nothing, temperature=1) -> indices

Draw `n` indices from each slice of `logits` along dimension `dims`, with probabilities
`softmax(logits ./ temperature; dims)`. This is the sampling step of text generation
from a language model, batched over sequences.

The returned `indices` have the size of `logits`, except along `dims` where they have
size `n`. Each index is a position along `dims`. For `n=1` the shape is that of
`argmax(logits; dims)`.

# Keyword arguments

- `dims`: the dimension of the categories.
- `n`: the number of draws per slice.
- `replace`: if `true`, the `n` draws are independent. If `false`, the `n` indices of a
  slice are distinct and come in the order of drawing them one at a time, each from the
  categories not drawn yet. Then `n` cannot exceed `size(logits, dims)`, and if fewer
  than `n` categories can be drawn (because of `k`, `p` or logits equal to `-Inf`), the
  remaining indices are excluded categories.
- `k`: if given, only the `k` largest logits of each slice can be drawn (top-k sampling).
- `p`: if given, only the smallest set of largest logits whose total probability
  reaches `p` can be drawn (top-p or nucleus sampling). The probabilities are computed
  after applying `temperature` and `k`.
- `temperature`: the logits are divided by `temperature` before sampling. Values below 1
  sharpen the distribution, values above 1 flatten it. `temperature=0` returns the
  indices of the largest logits (greedy decoding).

Ties at the `k`-th largest logit or at the top-`p` threshold are all kept.

The sample is drawn with the Gumbel-max trick, using only broadcasting, sorting and
reductions along `dims`, so it runs on GPU arrays without scalar indexing. Logits are
converted to at least `Float32` precision. This function is not differentiable.

With Reactant, pass an `rng` such as `Reactant.ReactantRNG()` to the compiled function.
Otherwise the seed is fixed at compile time and every call returns the same sample.

See also [`sample_categorical`](@ref) and [`topk`](@ref).

# Examples

```jldoctest
julia> logits = [1.0 5.0 0.0; 3.0 0.0 2.0; 2.0 1.0 9.0];  # 3 categories, 3 samples

julia> sample_logits(logits; temperature=0)
1×3 Matrix{Int64}:
 2  1  3

julia> sample_logits(logits; k=1)
1×3 Matrix{Int64}:
 2  1  3

julia> sample_logits(logits'; dims=2, k=1)
3×1 Matrix{Int64}:
 2
 1
 3

julia> sample_logits(logits; n=2, replace=false, temperature=0)
2×3 Matrix{Int64}:
 2  1  3
 3  3  2
```

A generation loop for a language model `model` that maps a `(seq_len, batch)` matrix
of tokens to `(vocab_size, seq_len, batch)` logits:

```julia
for _ in 1:max_new_tokens
    logits = model(tokens)[:, end, :]
    tokens = vcat(tokens, sample_logits(logits; k=50, temperature=0.8))
end
```
"""
sample_logits(logits::AbstractArray; kws...) = _sample_logits(nothing, logits; kws...)
sample_logits(rng::AbstractRNG, logits::AbstractArray; kws...) = _sample_logits(rng, logits; kws...)

function _sample_logits(rng, logits::AbstractArray; dims::Integer=1, n::Integer=1,
                        replace::Bool=true, k::Union{Nothing,Integer}=nothing,
                        p::Union{Nothing,Real}=nothing, temperature::Real=1)
    m = size(logits, dims)
    n >= 1 || throw(ArgumentError("n=$n must be positive"))
    replace || n <= m ||
        throw(ArgumentError("n=$n draws without replacement exceed size(logits, dims)=$m"))
    k === nothing || k >= 1 || throw(ArgumentError("k=$k must be positive"))
    p === nothing || 0 < p <= 1 || throw(ArgumentError("p=$p must be in the interval (0, 1]"))
    temperature >= 0 || throw(ArgumentError("temperature=$temperature must be non-negative"))
    x = _float32up(logits)
    if temperature == 0
        replace && n > 1 &&
            return repeat(_argmax_along(x, dims); outer=ntuple(d -> d == dims ? n : 1, ndims(x)))
        return _top_along(x, n, dims)
    end
    if temperature != 1
        x = x ./ eltype(x)(temperature)
    end
    k = k === nothing || k >= m ? nothing : k
    p = p === nothing || p == 1 ? nothing : p
    if k !== nothing || p !== nothing
        s = _sort(x, dims, true)
        if k !== nothing
            thr = minimum(ifelse.(_positions(s, dims) .<= k, s, typemax(eltype(s))); dims)
            x = _mask_below(x, thr)
            s = _mask_below(s, thr)
        end
        if p !== nothing  # keep a logit while the mass of the larger ones is below p
            q = NNlib.softmax(s; dims)
            before = cumsum(q; dims) .- q
            thr = minimum(ifelse.(before .< p, s, typemax(eltype(s))); dims)
            x = _mask_below(x, thr)
        end
    end
    # Gumbel-max: argmax(x .+ g) is a draw from softmax(x), and the n largest entries
    # of x .+ g are n draws without replacement.
    if replace && n > 1  # independent noise for each draw, along a new dimension dims + 1
        x = unsqueeze(x; dims = dims + 1)
        z = x .+ _gumbel(rng, x, Base.setindex(size(x), n, dims + 1))
        return dropdims(_argmax_along(z, dims); dims)
    end
    return _top_along(x .+ _gumbel(rng, x, size(x)), n, dims)
end

# Float16 and BFloat16 are too coarse for the Gumbel noise.
_float32up(x::AbstractArray) = promote_type(float(eltype(x)), Float32).(x)

function _gumbel(rng, x::AbstractArray, sz::Dims)
    # some RNGs (e.g. CURAND) draw from (0, 1] and u = 1 would give infinite noise
    u = min.(_rand_like(rng, x, sz), prevfloat(one(eltype(x))))
    return .-log.(.-log.(u))
end

_rand_like(::Nothing, x, sz) = rand_like(x, sz)
_rand_like(rng::AbstractRNG, x, sz) = rand_like(rng, x, sz)

_sort(x::AbstractVector, dims::Integer, rev::Bool) = sort(x; rev)
_sort(x::AbstractArray, dims::Integer, rev::Bool) = sort(x; dims, rev)

_mask_below(x, thr) = ifelse.(x .>= thr, x, typemin(eltype(x)))

# 1:n laid along `dims`, to broadcast against `x`. Use it in a fused broadcast
# with `x`, so that it is not materialized on the CPU when `x` is a GPU array.
function _positions(x::AbstractArray, dims::Integer)
    n = size(x, dims)
    return reshape(1:n, ntuple(d -> d == dims ? n : 1, ndims(x)))
end

# Position of the first maximum along `dims`. Not `argmax(x; dims)`, which returns
# CartesianIndices on CPU and GPU, but linear indices under Reactant.
_argmax_along(x, dims) =
    minimum(ifelse.(x .== maximum(x; dims), _positions(x, dims), typemax(Int)); dims)

# Positions of the n largest entries along `dims`, in decreasing order.
_top_along(x, n, dims) = n == 1 ? _argmax_along(x, dims) : last(topk(x, n; dims))

"""
    sample_categorical([rng], probs; dims=1, n=1, replace=true) -> indices

Draw `n` indices from each slice of `probs` along dimension `dims`, with probabilities
proportional to the entries of the slice. The entries must be non-negative, but they
need not sum to one.

The returned `indices` have the size of `probs`, except along `dims` where they have
size `n`. Equivalent to `sample_logits(rng, log.(probs); dims, n, replace)`, see
[`sample_logits`](@ref).

# Examples

```jldoctest
julia> probs = [0.0 0.0; 1.0 0.0; 0.0 3.0];

julia> sample_categorical(probs)
1×2 Matrix{Int64}:
 2  3
```
"""
sample_categorical(probs::AbstractArray; kws...) = _sample_categorical(nothing, probs; kws...)
sample_categorical(rng::AbstractRNG, probs::AbstractArray; kws...) =
    _sample_categorical(rng, probs; kws...)

_sample_categorical(rng, probs; dims::Integer=1, n::Integer=1, replace::Bool=true) =
    _sample_logits(rng, log.(_float32up(probs)); dims, n, replace)

@non_differentiable sample_logits(::Any...)
@non_differentiable sample_categorical(::Any...)
