struct SlidingWindow{T}
    data::T
    size::Int
    stride::Int
    count::Int
end

Base.length(A::SlidingWindow) = A.count

function Base.getindex(A::SlidingWindow, i::Int)
    1 <= i <= length(A) || throw(BoundsError(A, i))
    windowrange = getrange(A, i)
    return getobs(A.data, windowrange)
end

function getrange(A::SlidingWindow, i::Int)
    offset = 1 + (i-1) * A.stride
    return offset:offset+A.size-1
end

function Base.show(io::IO, A::SlidingWindow)
    return print(io, "slidingwindow($(A.data), size=$(A.size), stride=$(A.stride))")
end

Base.iterate(A::SlidingWindow, i::Int=1) = i > length(A) ? nothing : (A[i], i+1)

"""
    slidingwindow(data; size, stride=1) -> SlidingWindow

Return a vector-like view of the `data` for which each element is
a fixed size "window" of `size` adjacent observations. Note that only complete
windows are included in the output, which implies that it is
possible for excess observations to be omitted from the view.

Note that the windows are not materialized at construction time. 
To actually get a copy of the data at some window use indexing or [`getobs`](@ref).

```jldoctest
julia> s = slidingwindow(1:20, size=6)
slidingwindow(1:20, size=6, stride=1)

julia> s[1]
1:6

julia> s[2]
2:7
```

The optional parameter `stride` can be used to specify the
distance between the start elements of each adjacent window.
By default the stride is equal to 1.

```jldoctest
julia> s = slidingwindow(1:20, size=6, stride=3)
slidingwindow(1:20, size=6, stride=3)

julia> for w in s; println(w); end
1:6
4:9
7:12
10:15
13:18
```
"""
function slidingwindow(data; size::Int, stride::Int=1)
    size > 0 || throw(ArgumentError("Specified window size must be strictly greater than 0. Actual: $size"))
    size <= numobs(data) || throw(ArgumentError("Specified window size is too large for the given number of observations"))
    stride > 0 || throw(ArgumentError("Specified stride must be strictly greater than 0. Actual: $stride"))
    count = floor(Int, (numobs(data) - size + stride) / stride)
    return SlidingWindow(data, size, stride, count)
end

