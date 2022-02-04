# MLUtils.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaML.github.io/MLUtils.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaML.github.io/MLUtils.jl/dev)
[![](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://codecov.io/gh/JuliaML/MLUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaML/MLUtils.jl)

`MLUtils.jl` defines interfaces and implements common utilities for Machine Learning pipelines.

## Features

- An extensible dataset interface  (`numobs` and `getobs`).
- Data iteration and dataloaders (`eachobs` and `DataLoader`).
- Lazy data views (`obsview`). 
- Resampling procedures (`undersample` and `oversample`).
- Train/test splits (`splitobs`) 
- Data partitioning and aggregation tools (`batch`, `unbatch`, `chunk`, `group_counts`, `group_indices`).
- Folds for cross-validation (`kfolds`, `leavepout`).
- Datasets lazy tranformations (`mapobs`, `filterobs`, `groupobs`, `joinobs`, `shuffleobs`).
- Toy datasets for demonstration purpose. 
- Other data handling utilities (`flatten`, `normalise`, `unsqueeze`, `stack`, `unstack`).

## Related Packages

`MLUtils.jl` brings togheter functionalities previously found in [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) , [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl) and [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl). These packages are now discontinued. 

Other features were ported from the deep learning library [`Flux.jl`](https://github.com/FluxML/Flux.jl), as they are of general use. 

[` MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) is a more complete package for managing the whole machine learning pipeline if you are looking for a sklearn replacement.
