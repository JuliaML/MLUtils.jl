
# to accumulate indices as views instead of copies
_view(indices::AbstractRange, i::Int) = indices[i]
_view(indices::AbstractRange, i::AbstractRange) = indices[i]
_view(indices, i::Int) = indices[i] # to throw error in case
_view(indices, i) = view(indices, i)
