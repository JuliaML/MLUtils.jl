
# mapobs

struct MappedData{F,D} <: AbstractDataContainer
    f::F
    data::D
end

Base.show(io::IO, data::MappedData) = print(io, "mapobs($(data.f), $(summary(data.data)))")
Base.show(io::IO, data::MappedData{F,<:AbstractArray}) where {F} =
    print(io, "mapobs($(data.f), $(ShowLimit(data.data, limit=80)))")
Base.length(data::MappedData) = numobs(data.data)
Base.getindex(data::MappedData, idx::Int) = data.f(getobs(data.data, idx))
Base.getindex(data::MappedData, idxs::AbstractVector) = data.f.(getobs(data.data, idxs))


"""
    mapobs(f, data)

Lazily map `f` over the observations in a data container `data`.
```julia
data = 1:10
getobs(data, 8) == 8
mdata = mapobs(-, data)
getobs(mdata, 8) == -8
```
"""
mapobs(f, data) = MappedData(f, data)
mapobs(f::typeof(identity), data) = data


"""
    mapobs(fs, data)

Lazily map each function in tuple `fs` over the observations in data container `data`.
Returns a tuple of transformed data containers.
"""
mapobs(fs::Tuple, data) = Tuple(mapobs(f, data) for f in fs)


struct NamedTupleData{TData,F} <: AbstractDataContainer
    data::TData
    namedfs::NamedTuple{F}
end

Base.length(data::NamedTupleData) = numobs(getfield(data, :data))

function Base.getindex(data::NamedTupleData{TData,F}, idx::Int) where {TData,F}
    obs = getobs(getfield(data, :data), idx)
    namedfs = getfield(data, :namedfs)
    return NamedTuple{F}(f(obs) for f in namedfs)
end

Base.getproperty(data::NamedTupleData, field::Symbol) =
    mapobs(getproperty(getfield(data, :namedfs), field), getfield(data, :data))

Base.show(io::IO, data::NamedTupleData) =
    print(io, "mapobs($(getfield(data, :namedfs)), $(getfield(data, :data)))")

"""
    mapobs(namedfs::NamedTuple, data)

Map a `NamedTuple` of functions over `data`, turning it into a data container
of `NamedTuple`s. Field syntax can be used to select a column of the resulting
data container.

```julia
data = 1:10
nameddata = mapobs((x = sqrt, y = log), data)
getobs(nameddata, 10) == (x = sqrt(10), y = log(10))
getobs(nameddata.x, 10) == sqrt(10)
```
"""
function mapobs(namedfs::NamedTuple, data)
    return NamedTupleData(data, namedfs)
end

# filterobs

"""
    filterobs(f, data)

Return a subset of data container `data` including all indices `i` for
which `f(getobs(data, i)) === true`.

```julia
data = 1:10
numobs(data) == 10
fdata = filterobs(>(5), data)
numobs(fdata) == 5
```
"""
function filterobs(f, data; iterfn = _iterobs)
    return obsview(data, [i for (i, obs) in enumerate(iterfn(data)) if f(obs)])
end

_iterobs(data) = [getobs(data, i) for i = 1:numobs(data)]


# groupobs

"""
    groupobs(f, data)

Split data container data `data` into different data containers, grouping
observations by `f(obs)`.

```julia
data = -10:10
datas = groupobs(>(0), data)
length(datas) == 2
```
"""
function groupobs(f, data)
    groups = Dict{Any,Vector{Int}}()
    for i = 1:numobs(data)
        group = f(getobs(data, i))
        if !haskey(groups, group)
            groups[group] = [i]
        else
            push!(groups[group], i)
        end
    end
    return Dict(group => obsview(data, idxs) for (group, idxs) in groups)
end

# joinumobs

struct JoinedData{T,N} <: AbstractDataContainer
    datas::NTuple{N,T}
    ns::NTuple{N,Int}
end

JoinedData(datas) = JoinedData(datas, numobs.(datas))

Base.length(data::JoinedData) = sum(data.ns)

function Base.getindex(data::JoinedData, idx)
    for (i, n) in enumerate(data.ns)
        if idx <= n
            return getobs(data.datas[i], idx)
        else
            idx -= n
        end
    end
end

"""
    joinobs(datas...)

Concatenate data containers `datas`.

```julia
data1, data2 = 1:10, 11:20
jdata = joinumobs(data1, data2)
getobs(jdata, 15) == 15
```
"""
joinobs(datas...) = JoinedData(datas)

"""
    shuffleobs([rng], data)

Return a "subset" of `data` that spans all observations, but
has the order of the observations shuffled.

The values of `data` itself are not copied. Instead only the
indices are shuffled. This function calls [`obsview`](@ref) to
accomplish that, which means that the return value is likely of a
different type than `data`.

```julia
# For Arrays the subset will be of type SubArray
@assert typeof(shuffleobs(rand(4,10))) <: SubArray

# Iterate through all observations in random order
for x in eachobs(shuffleobs(X))
    ...
end
```

The optional parameter `rng` allows one to specify the
random number generator used for shuffling. This is useful when
reproducible results are desired. By default, uses the global RNG.
See `Random` in Julia's standard library for more info.

For this function to work, the type of `data` must implement
[`numobs`](@ref) and [`getobs`](@ref). See [`ObsView`](@ref)
for more information.
"""
shuffleobs(data) = shuffleobs(Random.GLOBAL_RNG, data)

function shuffleobs(rng::AbstractRNG, data)
    obsview(data, randperm(rng, numobs(data)))
end
