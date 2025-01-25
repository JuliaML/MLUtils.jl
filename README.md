# MLUtils.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaML.github.io/MLUtils.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaML.github.io/MLUtils.jl/dev)
[![](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaML/MLUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![](https://codecov.io/gh/JuliaML/MLUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaML/MLUtils.jl)

*MLUtils.jl* defines interfaces and implements common utilities for Machine Learning pipelines.

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


## Examples

Let us take a look at a hello world example to get a feeling for 
how to use this package in a typical ML scenario. 

```julia
using MLUtils

# X is a matrix of floats
# Y is a vector of strings
X, Y = load_iris()

# The iris dataset is ordered according to their labels,
# which means that we should shuffle the dataset before
# partitioning it into training- and test-set.
Xs, Ys = shuffleobs((X, Y))

# We leave out 15 % of the data for testing
cv_data, test_data = splitobs((Xs, Ys); at=0.85)

# Next we partition the data using a 10-fold scheme.
for (train_data, val_data) in kfolds(cv_data; k=10)

    # We apply a lazy transform for data augmentation
    train_data = mapobs(xy -> (xy[1] .+ 0.1 .* randn.(), xy[2]),  train_data)

    for epoch = 1:10
        # Iterate over the data using mini-batches of 5 observations each
        for (x, y) in eachobs(train_data, batchsize=5)
            # ... train supervised model on minibatches here
        end
    end
end
```

In the above code snippet, the inner loop for `eachobs` is the
only place where data other than indices is actually being
copied. In fact, while `x` and `y` are materialized arrays, 
all the rest are data views. 


## Historical Notes

*MLUtils.jl* brings together functionalities previously found in [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) , [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl) and [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl). These packages are now discontinued. 

Other features were ported from the deep learning library [Flux.jl](https://github.com/FluxML/Flux.jl), as they are of general use. 


## Alternatives and Related Packages

- [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/) is a more complete package for managing the whole machine learning pipeline if you are looking for a sklearn replacement.

- [NNlib.jl](https://github.com/FluxML/NNlib.jl) provides utility functions for neural networks.

- [TableTransforms.jl](https://github.com/JuliaML/TableTransforms.jl) contains transformations for tabular datasets.

- [DataAugmentation.jl](https://github.com/FluxML/DataAugmentation.jl). Efficient, composable data augmentation for machine and deep learning with support for n-dimensional images, keypoints and categorical masks.

