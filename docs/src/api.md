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
batchview
BatchView
datasubset
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
Datasets.load_line
Datasets.load_poly
Datasets.load_sin
Datasets.load_spiral
Datasets.make_sin
Datasets.make_spiral
Datasets.make_poly
```


