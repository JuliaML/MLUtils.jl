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
    for epoch = 1:100
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


A common task when training Convolutional Neural Networks for image representations 
is to apply random augmentations to the training data. These augmentations are often
operations such as flipping the image or applying Gaussian Blur. This example shows
how to lazily apply such transformations at the time where the batch is loaded using
[`Augmentor.jl`](https://evizero.github.io/Augmentor.jl/stable/).
When training the model, one commonly iterates over mini-batches of data and applies the
augmentations batch-wise. Here we show how `MLUtils.jl` allows to implement this using a
custom dataset.

First, we import the packages we are using. Besides `MLUtils`, we are using `Random` for random
number generation, `Augmentor` for the augmentations, and `ImageCore` to convert numerical arrays
into images. For more details on using images in the Julia eco-system see the 
[JuliaImages documentation](https://juliaimages.org/stable/tutorials/quickstart/).

```julia
using MLUtils
using Random
using Augmentor
using ImageCore
```

The first step is to define a custom [type](https://docs.julialang.org/en/v1/manual/types/) that defines our dataset:

```julia
struct my_dset{T}
    data_arr::T
    trf
end
```

The structure takes a type parameter `T`, for numerical image data this could be `Array{Float32}(4)`.
That is, we specify that the numerical base type is `Float32`. The four dimensions correspond to
width, height, channels, and number of observations. The field `trf` stores the transformation we will
apply to the images. No type parameter is provided here, which allows to be more general for the type of
transformations we will apply.

The data we operate on is a 4-dimensional numerical array, that represents a large collection of color images:

```julia
num_samples = 100
num_channels = 3
width = height = 28
d = randn(Float32, width, height, num_channels, num_samples)
```

Now we can define a composition of transformations we wish to apply to the data. In this example we 
compose a horizontal, vertical flip or no operation, followed by a gaussian blur. A complete
list of available augmentations in `Augmentor.jl` is provided [here](https://evizero.github.io/Augmentor.jl/stable/operations/).

```julia
pl = FlipX() * FlipY() * NoOp() |> GaussianBlur(3:2:5, 1f0:1f-1:2f0)
```

With the data and transformation in place, we can instantiate the dataset

```julia
ds = my_dset(d, pl)
```

To instantiate a `DataLoader`to iterate over this simple dataset we need to implement custom
`numobs` and `getobs` methods:

```julia
function MLUtils.getobs(dset::my_dset, ix::Int)
    obs = dset.data_arr[:, :, :, ix]                     # Fetch a single observation from the dataset
    obs_c = colorview(RGB, permutedims(obs, (3, 1, 2)))  # Convert it into an image so that the transformation can be applied to it
    obs_trf = augment(obs_c, dset.trf)                   # Apply the augmentations
    permutedims(channelview(obs_trf), (2, 3, 1))         # Convert the augmented observation into numerical data
end

MLUtils.numobs(data::my_dset) = size(data.data_arr)[end]
```

The `numobs` function just returns the number of samples in the dataset. Which is just the extend of the
last dimnension of the data array field of `my_dset`. The `getobs` function takes the dataset and an
integer index as input and return the augmented array. Internally, we first fetch a single observation
from the dataset. Then we convert it into an image, apply the augmentation, and convert the augmented
observation back into a numerical type.

With these methods implemented, we can now construct a `DataLoader` and iterate over the dataset.
The augmentations will be applied lazily at the time a observation is accessed.

```julia
loader = DataLoader(ds, batchsize=-1)

for (ix, obs) ∈ enumerate(loader)
    @show ix, size(obs)
end
```

Now we focus on batching. In practice we want to train on a batch of multiple images.
`MLUtils.jl` provides `BatchView` that allows to fetch batches of images at a time.
To make `BatchView` work on our dataset it needs to implement the data container
interface as described in `ObsView`. In particular, we need to implement a
`getobs` and `getobs!` method that fetch multiple observations.

The difference between `getobs!` and `getobs` is that `getobs!` returns multiple
observations in a pre-allocated buffer. We can therefore implement `getobs!` first
and let `getobs` allocate a buffer and just call `getobs!`.

```julia
function MLUtils.getobs!(buffer, dset, ix::AbstractVector)
    batch = dset.data_arr[:, :, :, ix]                           # Load selected observations
    batch_img = colorview(RGB, permutedims(batch, (3, 1, 2, 4))) # Convert to image
    augmentbatch!(CPUThreads(), buffer, batch_img, dset.trf)     # Augment entire batch
    permutedims(channelview(buffer), (2, 3, 1, 4))               # Convert augmented batch to numerical type
end

function MLUtils.getobs(dset::my_dset, ix::AbstractVector)
    # Get the size of the dataset array, sans the number of batches. THat is defined by 
    # the length of the index vector

    batch_dim = [size(ds.data_arr)[[3, 1, 2]]..., length(ix)]
    buffer = colorview(RGB, zeros(eltype(dset.data_arr), batch_dim...))
    MLUtils.getobs!(buffer, dset, ix)
end
```

`getobs!` takes as input the pre-allocated buffer, the dataset, and vector of
indices that specify the desired observations to fetch. The function then
copies the specified observations into a buffer and converts the datatype of the
buffer into an image type. Then, the entire batch is augmented and the result stored
in the pre-alloacted buffer. The first argument `CPUThreads()` in the call to
`augmentbatch!` allows individual augmentations to be performed in parallel.
The result is converted back into a numerical array which is then returned by the function.


The `getobs` method is essentially a wrapper around `getobs!` but also allocates
a buffer. The number of observations requested in each iteration of the `DataLoader`
can vary when there is a remainder for dividing number of total observations by the
batch size.

With these methods implemented, we can now lazily apply random augmentations to each
batch of the dataset:

```julia
loader_batch = DataLoader(ds, batchsize=27, shuffle=true)
for (ix, bobs) ∈ enumerate(loader_batch)
    @show ix, size(bobs)
end
```



## Related Packages

`MLUtils.jl` brings together functionalities previously found in [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) , [MLDataPattern.jl](https://github.com/JuliaML/MLDataPattern.jl) and [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl). These packages are now discontinued. 

Other features were ported from the deep learning library [`Flux.jl`](https://github.com/FluxML/Flux.jl), as they are of general use. 

[` MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/) is a more complete package for managing the whole machine learning pipeline if you are looking for a sklearn replacement.
