# MLUtils

[![Build Status](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaML/MLUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaML/MLUtils.jl)


This package embodies a community effort to provide common and extensible functionalities for Machine Learning packages in Julia.

The aim is to consolidate packages in the ML ecosystem such as [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl) and [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl) into a single well-mantained repository.

## Interface

### Observation

The functions `numobs`, `getobs`, and `getobs!` are implemented for basic types (arrays, tuples, ...)
and can be extend

For array types, the observation dimension is the last dimension. This means that when working 
with matrices each columns is considered as an individual observation. 

```    
    numobs(data)

Return the total number of observations contained in `data`.
See also [`getobs`](@ref)
```

```
    getobs(data, idx)

Return the observations corresponding to the observation-index `idx`.
Note that `idx` can be any type as long as `data` has defined
`getobs` for that type.
The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision themselves.
The output should be consistent when `idx` is a scalar vs vector.
See also [`getobs!`](@ref) and [`numobs`](@ref) 
```

```
    getobs!(buffer, data, idx)

Inplace version of `getobs(data, idx)`. If this method
is defined for the type of `data`, then `buffer` should be used
to store the result, instead of allocating a dedicated object.
Implementing this function is optional. In the case no such
method is provided for the type of `data`, then `buffer` will be
*ignored* and the result of `getobs` returned. This could be
because the type of `data` may not lend itself to the concept
of `copy!`. Thus, supporting a custom `getobs!` is optional
and not required.
```

