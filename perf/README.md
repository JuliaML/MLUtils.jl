# `DataLoader` data-loading benchmarks

Loader-only timing of MLUtils' multiprocess `DataLoader` (`num_workers`) and its shared-memory
batch transport (`shm`), with a PyTorch `DataLoader` run on the same task for comparison.

The workload is CIFAR-10 **on-the-fly** loading: images are pulled from the HuggingFace `datasets`
Arrow dataset and decoded + augmented (pad-4 random 32×32 crop + horizontal flip, per-channel
normalize) every batch, `batchsize = 128`. This is the case `num_workers` exists for — the CPython
image decode runs under the GIL, so it does not scale on threads and needs worker *processes*.

Each benchmark iterates the training loader with **no model**, isolating data loading (decode +
augment + collate + worker IPC) from GPU compute.

## Methodology: warm steady-state

The **first epoch** pays a large one-time cost: leasing/spawning worker processes and initializing a
Python interpreter (PythonCall / `datasets`) in each. That cost is amortized across a whole training
run — the pool is kept warm across every epoch and every loader — so it is reported separately from
the **warm** epochs. The warm per-epoch time is the real recurring data-loading cost, and is what the
speedups below are computed from.

## Running

```bash
# Julia (MLUtils; dev's ../ and, by default, a sibling HuggingFaceDatasets checkout)
julia --project=perf perf/bench_loader.jl        # EPOCHS=8 by default

# PyTorch (self-contained uv script; downloads torch/torchvision/datasets)
uv run perf/bench_loader.py                       # EPOCHS=8
```

## Results

Hardware: AMD Ryzen Threadripper PRO 9955WX (16C/32T), Julia 1.12, PyTorch on CPU loading (no GPU
used — loader-only). CIFAR-10 train, 50 000 images, `batchsize=128` → 391 batches/epoch, 8 epochs.

Column meanings:

- **setup** — wall time of the **first** epoch, which includes the one-time pool warmup (spawn the
  worker processes, init a Python interpreter in each, ship the dataset). Paid once per warm pool.
- **warm/epoch** — mean wall time of a **subsequent** epoch (mean of epochs 2–8), with the pool
  already warm. This is the recurring per-epoch data-loading cost during a training run.
- **speedup** — `serial warm/epoch ÷ this row's warm/epoch` (higher is faster).

### MLUtils (`num_workers` + `shm`)

| config            | setup — 1st epoch (s) | warm/epoch — mean of epochs 2–8 (s) | speedup vs serial |
|-------------------|-----------------------|-------------------------------------|-------------------|
| serial            | 3.4       | 3.34             | 1.00×             |
| `num_workers=2`, `shm=false` | 12.8 | 2.21       | 1.51×             |
| `num_workers=2`, `shm=true`  | 12.5 | 2.08       | 1.60×             |
| `num_workers=4`, `shm=false` | 12.2 | 1.33       | 2.51×             |
| `num_workers=4`, `shm=true`  | 12.3 | 1.08       | 3.08×             |
| `num_workers=8`, `shm=false` | 12.6 | 0.83       | 4.01×             |
| `num_workers=8`, `shm=true`  | 12.9 | 0.67       | **4.98×**         |

### PyTorch (`torch.utils.data.DataLoader`, `persistent_workers=True`)

| config          | setup — 1st epoch (s) | warm/epoch — mean of epochs 2–8 (s) | speedup vs serial |
|-----------------|-----------------------|-------------------------------------|-------------------|
| serial          | 6.3       | 6.36             | 1.00×             |
| `num_workers=2` | 3.5       | 3.44             | 1.85×             |
| `num_workers=4` | 1.9       | 1.80             | 3.52×             |
| `num_workers=8` | 1.1       | 0.99             | 6.45×             |

## Takeaways

- **`num_workers` scales the GIL-bound decode near-linearly** — MLUtils reaches ~5× at 8 workers, a
  PyTorch-shaped curve.
- **`shm` helps at every worker count, and its edge grows with workers**: at `num_workers=8` it cuts
  the warm epoch from 0.83 s to 0.67 s (~20% faster). As workers multiply, the per-batch *serialize
  → socket → deserialize* copy of the old transport becomes the bottleneck; `shm` replaces the bulk
  copy with a shared-memory handle, so only that config keeps scaling cleanly.
- **In absolute warm throughput MLUtils is competitive with — here, faster than — PyTorch**: 0.67 s
  vs 0.99 s per warm epoch at 8 workers. MLUtils' serial decode is also ~1.9× faster (3.34 s vs
  6.36 s), because the augment is vectorized over the batch in Julia rather than run per-sample in
  Python. PyTorch's larger *relative* speedup (6.45× vs 4.98×) mostly reflects its heavier serial
  baseline, i.e. more room to parallelize away.
- **MLUtils' one weak spot is the one-time pool warmup** (~13 s vs PyTorch's ~1 s): each Julia
  worker boots a full Python interpreter (`PythonCall` / `datasets`) and re-opens the dataset, which
  is much heavier than forking a CPython worker. It is paid once per warm pool and amortizes away
  over a real training run, but it makes a *short* run look like it barely beats serial — which is
  exactly why the table separates setup from warm steady-state.
