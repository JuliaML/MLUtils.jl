# Data-loading-only benchmark for MLUtils' distributed `DataLoader` (`num_workers`) and its
# shared-memory batch transport (`shm`), on the CIFAR-10 on-the-fly decode+augment pipeline.
#
#   julia --project=perf perf/bench_loader.jl          # EPOCHS=8 by default
#
# It iterates the training loader with NO model, isolating decode + augment + collate + worker IPC.
# The first epoch pays the one-time pool warmup (spawn workers, ship the dataset, init a Python
# interpreter per worker) and is reported separately from the WARM steady-state epochs — the latter
# is the real per-epoch loading cost, amortized across a whole training run.
using HuggingFaceDatasets
using MLUtils: MLUtils, mapobs, DataLoader

const CIFAR_MEAN = reshape(Float32[0.4914, 0.4822, 0.4465], 1, 1, 3, 1)
const CIFAR_STD  = reshape(Float32[0.2470, 0.2435, 0.2616], 1, 1, 3, 1)

function cifar_decode(batch)
    x = Float32.(batch["img"]) ./ 255f0       # (C, W, H, N)
    x = permutedims(x, (2, 3, 1, 4))          # (W, H, C, N)
    x = (x .- CIFAR_MEAN) ./ CIFAR_STD
    return (; image = x, label = batch["label"])
end

function cifar_augment(batch)                  # zero-pad by 4, random 32x32 crop, random h-flip
    x = batch.image
    W, H, C, N = size(x)
    pad = 4
    out = similar(x)
    padded = zeros(Float32, W + 2pad, H + 2pad, C)
    for n in 1:N
        fill!(padded, 0f0)
        @views padded[pad+1:pad+W, pad+1:pad+H, :] .= x[:, :, :, n]
        i, j = rand(0:2pad), rand(0:2pad)
        if rand(Bool)
            @views out[:, :, :, n] .= padded[i+W:-1:i+1, j+1:j+H, :]
        else
            @views out[:, :, :, n] .= padded[i+1:i+W, j+1:j+H, :]
        end
    end
    return (; image = out, label = batch.label)
end

cifar_train_transform(batch) = cifar_augment(cifar_decode(batch))

# Return per-epoch wall times for one loader configuration.
function epoch_times(train_data; epochs, num_workers, shm, batchsize=128)
    dl = DataLoader(train_data; batchsize, shuffle=true, num_workers, shm)
    times = Float64[]
    for _ in 1:epochs
        push!(times, @elapsed (for (x, y) in dl; end))
    end
    MLUtils.close_dataloader_pool()
    return times
end

const EPOCHS = parse(Int, get(ENV, "EPOCHS", "8"))

train_ds = load_dataset("uoft-cs/cifar10", split = "train")
train_data = mapobs(cifar_train_transform, train_ds)

println("### WARMUP (precompile both paths)")
epoch_times(train_data; epochs=1, num_workers=0, shm=nothing)
epoch_times(train_data; epochs=2, num_workers=2, shm=true)
MLUtils.close_dataloader_pool()

results = Tuple{String,Float64,Float64}[]   # (label, setup = epoch 1, warm = mean(epoch 2:end))
function bench(label, num_workers, shm)
    MLUtils.close_dataloader_pool(); GC.gc()
    ts = epoch_times(train_data; epochs=EPOCHS, num_workers, shm)
    warm = length(ts) > 1 ? sum(ts[2:end]) / (length(ts) - 1) : ts[1]
    println("### $label:  epoch1(cold)=$(round(ts[1],digits=2))s   warm/epoch=$(round(warm,digits=3))s")
    push!(results, (label, ts[1], warm))
    return warm
end

println("\n#### CIFAR-10 loader-only, on-the-fly decode+augment, $EPOCHS epochs ###########")
bench("serial          ", 0, false)
for nw in (2, 4, 8)
    bench("nw=$nw shm=false", nw, false)
    bench("nw=$nw shm=true ", nw, true)
end

println("\n#### SUMMARY — WARM steady-state per-epoch (the real loading cost) ###########")
serial_warm = results[1][3]
println(rpad("config", 18), rpad("setup (s)", 12), rpad("warm/epoch (s)", 16), "speedup")
for (label, setup, warm) in results
    println(rpad(strip(label), 18), rpad(round(setup, digits=2), 12),
            rpad(round(warm, digits=3), 16), string(round(serial_warm / warm, digits=2), "×"))
end
