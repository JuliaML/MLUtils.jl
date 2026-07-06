```@meta
DocTestSetup = quote
    using MLUtils
end
```

# Data Loaders

The [`DataLoader`](@ref) is the object you reach for in training loops. It wraps a
data container and turns it into an iterator over mini-batches, with shuffling,
partial-batch handling, custom collation, and — when loading is the bottleneck —
parallel or distributed loading, all through keyword arguments.

```julia
Xtrain = rand(10, 100)
Ytrain = rand('a':'z', 100)

loader = DataLoader((data=Xtrain, label=Ytrain); batchsize=5, shuffle=true)

for epoch in 1:100
    for (x, y) in loader        # destructure the named tuple
        @assert size(x) == (10, 5)
        @assert size(y) == (5,)
        # update the model on this mini-batch
    end
end
```

A `DataLoader` is built on the same [`numobs`](@ref)/[`getobs`](@ref) interface as
everything else in MLUtils, so it works with plain arrays, tuples, named tuples,
and any custom [data container](@ref "Data Containers").

## Key keyword arguments

- **`batchsize`**: observations per mini-batch. If negative, iterate over single
  observations. Default `1`.
- **`shuffle`**: reshuffle the observations *at the start of every epoch*. This
  differs from wrapping the data in [`shuffleobs`](@ref), which fixes a single
  permutation for the lifetime of the view. Default `false`.
- **`partial`**: when the number of observations is not divisible by
  `batchsize`, keep (`true`) or drop (`false`) the smaller last batch. Default
  `true`.
- **`collate`**: how a vector of observations is combined into a batch (see
  [Collation](@ref)). Default `nothing`.
- **`parallel`**: load batches on worker **threads**. Default `false`.
- **`num_workers`**: load batches on worker **processes**. Default `0` (off).
- **`buffer`**: reuse a preallocated buffer across iterations via
  [`getobs!`](@ref), avoiding per-batch allocations. Default `false`.

See the [`DataLoader`](@ref) docstring for the complete list. The
[`eachobs`](@ref) function is a thin convenience wrapper that constructs a
`DataLoader` for a quick iteration (see [Iteration & Views](@ref)).

## Collation

By default a batch is formed by `getobs(data, indices)`. The `collate` keyword
(shared with [`BatchView`](@ref)) lets you change this:

- `collate=nothing` (default): the batch is `getobs(data, indices)`.
- `collate=false`: the batch is the *vector* `[getobs(data, i) for i in indices]`,
  leaving the observations uncombined.
- `collate=true`: apply [`batch`](@ref) to that vector, recursively stacking
  arrays along a new last dimension.
- `collate=f`: use your own function `f` on the vector of observations.

A custom collate function is the idiomatic way to handle observations of varying
size — for example, padding variable-length sequences into a single padded batch:

```jldoctest
julia> loader = DataLoader(["a", "bb", "ccc", "dddd"]; batchsize=2, collate=join);

julia> first(loader)
"abb"
```

## Parallel vs. distributed loading

Loading is worth parallelizing only when `getobs` is *expensive* — reading and
decoding an image from disk, running a heavy augmentation, calling into Python. If
`getobs` just indexes an in-memory array, the coordination overhead outweighs any
gain and plain **serial** loading is fastest. When loading *is* the bottleneck,
MLUtils offers two strategies:

- **`parallel=true`** spreads `getobs` over **threads** within the current
  process. Threads share memory, so the data container is never copied, and there
  is no per-batch inter-process communication. This is the right choice when
  `getobs` is pure-Julia and CPU-bound. It requires starting Julia with multiple
  threads (`julia -t auto`, or `Threads.nthreads() > 1`).

- **`num_workers=N`** spreads `getobs` over `N` separate worker **processes** (via
  `Distributed`), mirroring PyTorch's `DataLoader(num_workers=N)`. Each process
  has its own memory space and — crucially — its own Python interpreter. This is
  the right choice when `getobs` does *not* scale under threads: most sharply a
  [`PythonCall`](https://github.com/JuliaPy/PythonCall.jl)-backed container, where
  CPython's global interpreter lock (GIL) lets only one thread run Python at a
  time, capping threaded loading at ~1×.

The two are **mutually exclusive** — set at most one. Both **break ordering
guarantees**: batches arrive in whatever order the workers finish them (exactly as
in PyTorch), so do not rely on batch order.

| | `parallel=true` (threads) | `num_workers=N` (processes) |
|---|---|---|
| Memory | shared — no copy of `data` | `data` serialized to each worker once |
| Startup cost | negligible | spawns processes (amortized: kept warm) |
| Per-batch overhead | none (shared memory) | ship index set out; collated batch back through **shared memory** (Unix) or serialized |
| GIL-bound `getobs` (Python) | no speedup (one shared GIL) | scales (one GIL per process) |
| Pure-Julia CPU-bound `getobs` | scales | scales, but pays IPC + copy |
| Large in-memory `data` | ideal (zero-copy) | costly (copied per worker) |
| Requirements | `julia -t N` | `data`/observations serializable |

Rule of thumb: **threads for pure-Julia CPU work, processes for GIL-bound or
otherwise non-thread-safe work.** The two worked examples below take one each.

### The distributed container contract: be serializable

`num_workers` uses only `getobs`/`numobs`, exactly as before. The one extra
requirement is that the `data` container — and the values `getobs` returns — be
serializable with Julia's stdlib `Serialization`. This is the direct analog of
PyTorch requiring a *picklable* dataset for its spawned workers. Arrays, tuples and
named tuples already satisfy this, so they work unchanged. A container holding a
non-serializable handle (a `PythonCall.Py`, a live database connection) must define
a `Serialization.serialize`/`deserialize` pair that ships a cheap *recipe* and
rebuilds the handle on the worker — again, exactly as a PyTorch dataset owns its
pickling behavior. `HuggingFaceDatasets` does this for you (see below).

Under the hood MLUtils sends `data` to each worker **once** (via a
`Distributed.CachingPool`), then dispatches only the per-batch index sets; `getobs`
and `collate` run on the worker so a single collated array is shipped back per
batch. The pool is started lazily on the first distributed iteration, kept warm
across loaders and epochs, and — being child processes — terminated automatically
when Julia exits. To release the workers earlier, call
`MLUtils.close_dataloader_pool()`.

### Shared-memory batch transport

Returning each collated batch by *serializing* it, pushing the bytes through a
socket, and deserializing on the main process is a full bulk copy per batch — and
for large batches it becomes the scaling bottleneck, so that adding workers may not
beat serial loading. To avoid it, on Unix MLUtils moves batches through **shared
memory**, the same trick as `torch.multiprocessing`: the worker writes each large
array leaf of a batch into a memory-mapped file (under `/dev/shm` on Linux,
`tempdir()` on other Unix) and only a small *handle* crosses the socket; the main
process maps the file back to an array and immediately unlinks it. Because the
pool's workers are always local processes on the same host, they share the
filesystem, so no bytes are copied across the socket. This is automatic — there is
no option to configure. Windows cannot unlink a memory-mapped file, so it uses the
serialized path. Two properties worth knowing:

- Received array leaves are dense, **writable** `Array`s. Views, `BitArray`s and
  other non-`Array` arrays are normalized to a dense `Array`, and a container's
  element type may widen (e.g. a `Dict{String,Any}`).
- Batches are backed by shared memory until they are garbage-collected, so avoid
  holding many epochs of batches alive at once (their pages are invisible to Julia's
  GC accounting, and `/dev/shm` can be small — 64 MB by default in some containers).

!!! note "Iterate Python-backed loaders from the main thread"
    For a `PythonCall`-backed loader, the container is serialized on the task that
    first iterates it. Iterate such a loader from the thread that can call Python
    (in practice, thread 1) — not from inside a `Threads.@spawn`.

## Worked example: threaded image augmentation (`parallel`)

A common task when training convolutional networks is to apply random augmentations
— flips, rotations, crops — to the training images. The augmentations should be
applied *lazily*, at the moment a batch is loaded, so each epoch sees freshly
perturbed data and nothing is materialized ahead of time. Augmentation is
pure-Julia and CPU-bound, so this is a textbook case for `parallel=true`.

We use [`DataAugmentation.jl`](https://github.com/FluxML/DataAugmentation.jl) for
the pipeline and [`ImageCore`](https://juliaimages.org/stable/) to convert between
numerical arrays and images.

```julia
using MLUtils
using DataAugmentation
using ImageCore
```

We define a type holding the data and the transformation pipeline. The data is a
4-dimensional array of color images following the convention that the last
dimension is the observation dimension; the axes are width, height, channels, and
number of observations. Both fields are type parameters so the struct stays
concretely typed and fast:

```julia
struct ImageDataset{T,F}
    data::T          # width × height × channels × observations
    transform::F     # a DataAugmentation.jl pipeline
end
```

For this example we generate a random batch of 100 RGB images of size 28×28 and
compose the augmentation pipeline with `|>`. `Maybe(tfm)` applies `tfm` with
probability `0.5`, `Rotate(15)` draws an angle uniformly from `[-15°, 15°]`, and
the trailing crop keeps every observation the same size so they can be stacked:

```julia
num_samples, num_channels, width, height = 100, 3, 28, 28
data = rand(Float32, width, height, num_channels, num_samples)

pipeline = Maybe(FlipX{2}()) |>
           Maybe(FlipY{2}()) |>
           Rotate(15) |>
           RandomResizeCrop((width, height)) |>
           ImageToTensor()

dataset = ImageDataset(data, pipeline)
```

To turn `ImageDataset` into a data container we make it indexable: [`numobs`](@ref)
and [`getobs`](@ref) fall back to `Base.length` and `Base.getindex`, so defining
those is enough. `getindex` fetches a single observation, wraps it as an `Image` so
the pipeline can be applied, and extracts the resulting tensor with `itemdata`:

```julia
Base.length(d::ImageDataset) = size(d.data, 4)

function Base.getindex(d::ImageDataset, i::Int)
    obs = d.data[:, :, :, i]                            # width × height × channels
    img = colorview(RGB, permutedims(obs, (3, 2, 1)))  # to an RGB image
    item = Image(img)
    return itemdata(apply(d.transform, item))          # augment, back to a numerical tensor
end
```

(You could equivalently define `MLUtils.numobs`/`MLUtils.getobs` directly; the
indexing interface is just the more familiar fallback.)

Since we only defined indexing for a single observation, we ask the `DataLoader` to
assemble batches with `collate=true`, which stacks the per-observation results with
[`batch`](@ref). Passing `parallel=true` overlaps the (CPU-bound) augmentation with
training on worker threads — start Julia with `julia -t auto` to benefit:

```julia
loader = DataLoader(dataset; batchsize=27, shuffle=true, collate=true, parallel=true)

for (i, x) in enumerate(loader)
    @show i, size(x)        # (28, 28, 3, 27) for the full batches
end
```

For allocation-free loading, DataAugmentation's
[`Buffered`](https://fluxml.ai/DataAugmentation.jl/dev/) pipelines pair naturally
with a [`getobs!`](@ref) method and the `buffer=true` option; see
[Data Containers](@ref) for details.

## Worked example: a Hugging Face dataset (`num_workers`)

[`HuggingFaceDatasets.jl`](https://github.com/JuliaGenAI/HuggingFaceDatasets.jl)
exposes the Python [🤗 `datasets`](https://huggingface.co/docs/datasets) library as
MLUtils data containers. Reading a sample calls into Python, so threaded loading is
capped by the GIL — this is the case `num_workers` exists for. Because a `Dataset`
defines the `serialize`/`deserialize` recipe described above (it pickles
by *reference* to its on-disk Arrow files and re-mmaps on the worker), it ships to
workers cheaply and needs no special handling on your part.

```julia
using MLUtils, HuggingFaceDatasets

ds = load_dataset("ylecun/mnist", split="train")   # 60_000 observations, read lazily
```

Each observation is read in the default `"julia"` format, so `getobs(ds, i)`
returns a plain `Dict` (here with keys `"image"` and `"label"`). Compose a
per-sample preprocessing step with [`mapobs`](@ref), exactly as for any other data
container — `num_workers` handles the nested `Dataset` transparently, auto-loading
`HuggingFaceDatasets` on the workers so they can reconstruct it:

```julia
function preprocess(sample)
    return (x = Float32.(sample["image"]) ./ 255, y = sample["label"])
end

loader = DataLoader(mapobs(preprocess, ds);
                    batchsize=128, shuffle=true, num_workers=4, collate=true)

for (x, y) in loader
    # each batch is assembled by one of 4 worker processes, each with its own
    # Python interpreter — so the four reads run in parallel, not GIL-serialized
end
```

This scales near-linearly in the number of workers, up to disk/CPU limits, where a
threaded `parallel=true` loader over the same Python-backed dataset would not speed
up at all.

!!! note "Named functions and script-local structs are shipped for you"
    A `num_workers` loader serializes your `data`/`collate` to each worker process.
    Julia ships an **anonymous** closure whole (by value), but a **named** function or
    a custom `struct` goes *by reference* — the worker needs that definition to exist.
    MLUtils bridges this automatically: it walks `data`/`collate`, recovers the
    **source** of every referenced `Main` function and struct (transitive helpers, the
    globals they read, a struct container's `getobs`/`numobs` methods, and values a
    closure captures) and defines them on the workers before loading. So `preprocess`
    can be an ordinary named function, and a script-local `struct MyDataset … end` used
    as the container works too — no `@everywhere` needed.

    The one requirement is that these definitions live in a **file** (a script you run,
    or a package you load) so their source can be read. An object defined directly at
    the **REPL** or via `eval` has no source file; a loader that needs to ship it fails
    fast with a clear error naming it, rather than a cryptic worker-side `UndefVarError`
    — put it in a file, or fall back to an anonymous function.

## Where to go next

- [Iteration & Views](@ref) — `eachobs`, `BatchView`, sliding windows, and the
  batching primitives `DataLoader` is built on.
- [Data Containers](@ref) — the `numobs`/`getobs` interface all of this rests on.
- [API Reference](@ref) — the complete list of exported functions.
