
# AbstractDataContainer

# Having an AbstractDataContainer allows to define sensible defaults
# for Base (or other) interfaces based on our interface.
# Now mainly provides the iteration interface.
# This makes it easier for developers by reducing boilerplate.

abstract type AbstractDataContainer end

Base.size(x::AbstractDataContainer) = (numobs(x),)
Base.iterate(x::AbstractDataContainer, state = 1) =
    (state > numobs(x)) ? nothing : (getobs(x, state), state + 1)
Base.lastindex(x::AbstractDataContainer) = numobs(x)
Base.firstindex(::AbstractDataContainer) = 1

# Generic fallback for subtypes that don't define their own `show`.
# Concrete subtypes with a more specific `show` method take precedence.
Base.show(io::IO, x::AbstractDataContainer) =
    print(io, numobs(x), "-element ", nameof(typeof(x)))
