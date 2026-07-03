# End-to-end test of distributed (`num_workers`) data loading over a real `PythonCall`-backed
# container: a HuggingFaceDatasets `Dataset`. This is the motivating case for `src/distributed.jl`'s
# Change 1 (auto-load the container's package on workers, even nested inside `mapobs`) and
# Change 2 (serialize the container on the consumer's task, not the GIL-less feeder).
#
# A `Dataset` serializes by-reference to its on-disk Arrow files (HuggingFaceDatasets'
# `serialize(::Dataset)`); the worker re-mmaps and reads it under its own GIL. The nested
# `DataLoader(mapobs(f, ds); num_workers=N)` case only works with both changes: Change 1 loads
# `HuggingFaceDatasets` on the worker to deserialize the `Dataset` buried in a `MappedData`, and
# Change 2 keeps the pickle call on thread 1.
#
# Loading `HuggingFaceDatasets` already pulls in the whole PythonCall/CondaPkg stack, so the
# marginal cost of the cross-process part (two workers, each booting a Python interpreter) is
# small — it runs everywhere, giving real end-to-end coverage of the distributed path.

import HuggingFaceDatasets
using HuggingFaceDatasets: Dataset, load_from_disk

# `mapobs` target: pull a single scalar column out of a row so batches are plain integer
# vectors we can compare order-agnostically. `getobs(ds, i)` returns a row dict keyed by
# column name.
label_col(row) = row["label"]

@testset "num_workers over a HuggingFaceDatasets Dataset" begin
    # Change 1 sees `HuggingFaceDatasets` through the `MappedData` wrapper (it is the outermost
    # type's package only for the bare-`Dataset` case; nested, it is buried in a field).
    ds = Dataset((; x = reshape(collect(1:8*20), 8, 20), label = collect(0:19)))
    @test HuggingFaceDatasets in MLUtils._referenced_modules(mapobs(label_col, ds))

    MLUtils.close_dataloader_pool()

    # in-memory Dataset (materialized to a temp Arrow dir by `serialize`), wrapped in mapobs;
    # `label_col` is a named `Main` function, so it also exercises the worker-defs source-shipping
    dl = DataLoader(mapobs(label_col, ds); batchsize=5, num_workers=2, collate=true)
    @test length(dl) == 4
    labels = reduce(vcat, collect(dl))
    @test sort(labels) == collect(0:19)          # every row seen once, read on a worker

    # on-disk Dataset (pickles by reference to its Arrow files — the zero-copy path)
    dir = mktempdir()
    ds.save_to_disk(dir)
    dsdisk = load_from_disk(dir)
    dl2 = DataLoader(mapobs(label_col, dsdisk); batchsize=4, num_workers=2, shuffle=true, collate=true)
    @test sort(reduce(vcat, collect(dl2))) == collect(0:19)

    # a bare `Dataset` (no `mapobs`) also loads: batches are row dicts collated per column
    dl3 = DataLoader(ds; batchsize=10, num_workers=2, collate=false)
    got = sort(reduce(vcat, [b[k]["label"] for b in collect(dl3) for k in 1:length(b)]))
    @test got == collect(0:19)

    MLUtils.close_dataloader_pool()
end
