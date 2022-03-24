# API Reference

## Index

```@index
Order = [:type, :function]
Modules = [MLUtils]
Pages = ["api.md"]
```

## Docs

```@autodocs
Modules = [MLUtils]
Pages   = ["utils.jl"]
Private = false
```

```@docs
batch
batchsize
BatchView
DataLoader
eachobs
filterobs
getobs
getobs!
joinobs
groupobs
kfolds
leavepout
mapobs
numobs
obsview
ObsView
oversample
MLUtils.rpad
randobs
rpad(::AbstractVector, ::Integer, ::Any)
shuffleobs
splitobs
stack
unbatch
undersample
unsqueeze
unstack
```


## Datasets Docs

```@docs
Datasets.load_iris
Datasets.make_sin
Datasets.make_spiral
Datasets.make_poly
```


