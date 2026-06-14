# MLUtils.jl — agent guide

MLUtils.jl is the data-handling layer of the JuliaML/Flux ecosystem. It defines the
observation interface (`numobs` / `getobs` / `getobs!`), lazy data views, `DataLoader`,
and utilities for batching, splitting, folds, and resampling.

## Running code & tests

- **Use the Julia language server / `julia` MCP server if one is available** (e.g.
  `mcp__julia__julia_eval`) instead of spawning `julia` from the shell. It keeps a
  persistent REPL with Revise loaded, so edits to function bodies are picked up
  automatically — no restart needed. **Redefining a `struct` (new fields or type
  parameters) does require a restart**, since Revise cannot redefine types.
- Run the full test suite the canonical way:
  ```julia
  import Pkg; Pkg.test("MLUtils")
  ```
  This activates `test/Project.toml` (CUDA, Zygote, DataFrames, Transducers, …),
  which are **separate from the package's lean runtime deps** in the top-level
  `Project.toml`.
- **Test files are not standalone.** `test/runtests.jl` defines shared fixtures
  (`X`, `Y`, `CustomType`, `CustomArrayNoView`, `CustomRangeIndex`, …) and `using`s
  the test deps before `include`ing each file. `include("test/dataloader.jl")` on its
  own will throw `UndefVarError`. To iterate on one testset, run the whole suite, or
  replicate `runtests.jl`'s preamble before including the file.
- **Threading:** the parallel data-loading path (`parallel=true`, `eachobsparallel`)
  only does real work when `Threads.nthreads() > 1`. Start Julia with `-t auto` (or set
  `JULIA_NUM_THREADS`) to actually exercise it; otherwise it runs sequentially.
- **Doctests** in docstrings (`jldoctest` blocks) run during the docs build via
  Documenter (`docs/make.jl`), not during `Pkg.test`.
- CI (`.github/workflows/CI.yml`) tests Julia `min` (the compat floor, currently
  **1.10**), latest `1`, and `nightly`. Don't use syntax newer than 1.10.

## The core interface

Everything is built on three functions, imported and re-exported from
[MLCore.jl](https://github.com/JuliaML/MLCore.jl):

- `numobs(data)` — number of observations.
- `getobs(data, idx)` — materialize observation(s) at `idx` (Int or vector).
- `getobs!(buffer, data, idx)` — in-place version for buffered loading (optional).

A type becomes a usable data container by implementing `numobs` + `getobs` (and
`getobs!` if it should support `buffer=true`). Subtyping `AbstractDataContainer`
(`src/datacontainer.jl`) provides Base iteration/indexing defaults for free.

**Views are lazy.** `ObsView` and `BatchView` describe a slicing of the data without
copying; the actual data is produced by `getobs`. `DataLoader` wraps its input in an
`ObsView`, then a `BatchView`, then iterates (optionally in parallel and/or with
reused buffers).

### Collation (a subtle spot — `src/batchview.jl`, `src/dataloader.jl`)

`BatchView`'s `collate` field controls how per-observation results become a batch:

- `Val(nothing)` (default) — batch via `getobs(data, range)` directly.
- `Val(false)` — return a `Vector` of the individual observations, uncollated.
- `true` / a function — apply the collate function (`true` ⇒ `MLUtils.batch`) to the
  vector of observations. The result type generally **differs** from the per-obs
  buffer type; code on the buffered path must account for that.

## Source layout

| File | Contents |
|---|---|
| `src/MLUtils.jl` | module root; `include` order and all exports |
| `src/datacontainer.jl` | `AbstractDataContainer` + Base iteration defaults |
| `src/obsview.jl` | `ObsView` — lazy per-observation view |
| `src/batchview.jl` | `BatchView` — lazy batched view + collate logic |
| `src/dataloader.jl` | `DataLoader`, `eachobs` |
| `src/parallel.jl` | threaded loading: `eachobsparallel`, `Loader`, `RingBuffer` |
| `src/obstransform.jl` | lazy transforms: `mapobs`, `filterobs`, `groupobs`, `joinobs`, `shuffleobs` |
| `src/batch.jl` | `batch`, `batchseq`, `batch_sequence`, `unbatch` |
| `src/splitobs.jl` | `splitobs` (incl. stratified) |
| `src/folds.jl` | `kfolds`, `leavepout`, `timeseries_kfolds` |
| `src/resample.jl` | `oversample`, `undersample` |
| `src/slidingwindow.jl` | `slidingwindow` |
| `src/randobs.jl` | `randobs` |
| `src/utils.jl` | array utils: `chunk`, `flatten`, `normalise`, `rescale`, `unsqueeze`, `stack`, `unstack`, `*_like`, … |
| `src/Datasets/` | toy datasets (`load_iris`) |
| `src/deprecations.jl` | deprecated names |

`include` order matters for type references: `parallel.jl` is loaded after
`batchview.jl`, so it can dispatch on `BatchView`, but not vice-versa.

## Conventions

- Match the surrounding style; gradients/AD go through ChainRulesCore `rrule`s — test
  them with `test_zygote` (`test/test_utils.jl`).
- When you add or change a data-container behavior, add a `@testset` to the matching
  `test/<file>.jl` (and use existing fixtures from `runtests.jl` rather than redefining
  them). Add a regression test referencing the issue number for bug fixes.
- Keep the top-level `Project.toml` deps minimal; anything test-only belongs in
  `test/Project.toml`.
