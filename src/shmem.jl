# Shared-memory batch transport for the distributed `DataLoader` (`num_workers > 0`).
#
# The default `num_workers` path returns each collated batch from a worker process to the main
# process by *serializing* it, pushing the bytes through a socket, and deserializing — a full
# bulk copy per batch, funneled through the single main process. For large batches this is the
# scaling bottleneck (a worker count never beats serial). PyTorch sidesteps it by passing tensors
# through shared memory: only a small *handle* crosses the socket, never the bulk bytes.
#
# This file adds the same idea for generic Julia batches. Because the pool's workers are always
# local `addprocs` on the same host, main and workers share a filesystem — on Linux `/dev/shm`
# (tmpfs, RAM-backed), on other Unix `tempdir()`. A worker writes each large array leaf of a
# batch into a memory-mapped file and replaces it, in the batch's structure, with a tiny
# `ShmHandle`; that skeleton is what crosses `remotecall_fetch`. The main side maps each handle
# back to an `Array`, immediately `rm`s the backing file (POSIX keeps the pages alive until the
# array is garbage-collected and munmapped), and hands the reconstructed batch to the consumer.
#
# There are no `Serialization` hooks: packing and unpacking are explicit two-sided transforms,
# which keeps failure handling and cleanup in plain, readable code.
#
# Windows cannot unlink a file while it is memory-mapped, so the whole mechanism is Unix-only
# (`_shm_supported()`); on Windows the loader silently uses the serialized path.

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A per-loader shared-memory session: a private directory holding this loader's leaf files, plus
# the size threshold (snapshotted from `_SHM_MIN_BYTES` at creation, see `_shm_session_create`).
# Plain and serializable, so the worker closure can carry it across the wire and write into `dir`.
struct ShmSession
    dir::String
    minbytes::Int
end

# Replaces a large array leaf in a packed batch skeleton. Carries only the backing file path and
# the array shape; `T`/`N` recover the element type and dimensionality for `Mmap.mmap`.
struct ShmHandle{T,N}
    file::String
    dims::NTuple{N,Int}
end

# A packed `Vector` of non-bits elements (e.g. a `collate=false` vector-of-observations): each
# element is packed recursively into `items`. Unpack rebuilds a `Vector{E}` when the elements
# still fit `E` (see `_shm_unpack`).
struct ShmVector{E}
    items::Vector{Any}
end

# A packed `AbstractDict{K,V}`: values packed recursively, keys ride along verbatim. Unpack
# rebuilds `Dict{K,V}` when every value still `isa V`, else falls back to `Dict{K,Any}` (never
# silently narrows a `Dict{String,Any}`).
struct ShmDict{K,V}
    pairs::Vector{Pair{K,Any}}
end

# ---------------------------------------------------------------------------
# Configuration & platform gating
# ---------------------------------------------------------------------------

# Array leaves at least this many bytes are moved through shared memory; smaller leaves ride the
# serialized skeleton (the handle + a syscall would cost more than copying them). A `Ref` so tests
# can lower it; snapshotted into each session at creation, so changing it does not affect a loader
# that has already built its session.
const _SHM_MIN_BYTES = Ref(16 * 1024)

# Recursion-depth guard for the pack/unpack walk over nested containers.
const _SHM_MAX_DEPTH = 32

# Shared memory across unlink-while-mapped is a POSIX guarantee; Windows lacks it.
_shm_supported() = Sys.isunix()

# Directory holding all sessions: Linux `/dev/shm` (tmpfs, RAM-backed) when present, else the
# platform temp dir (macOS and other Unix).
function _shm_root()
    shm = "/dev/shm"
    return (Sys.islinux() && isdir(shm)) ? shm : tempdir()
end

# ---------------------------------------------------------------------------
# Session lifecycle & cleanup
# ---------------------------------------------------------------------------

const _SHM_LOCK = ReentrantLock()
const _SHM_SESSIONS = String[]              # registry of live session dirs (main process only)
const _SHM_COUNTER = Ref(0)                 # disambiguates same-pid sessions
const _SHM_ATEXIT_REGISTERED = Ref(false)

# Is process `pid` still alive? Uses `kill(pid, 0)`: 0 → alive; `EPERM` → alive but not ours
# (still counts as alive, do not sweep); `ESRCH` → dead. Non-Unix: conservatively "alive".
function _pid_alive(pid::Int)
    Sys.isunix() || return true
    ret = ccall(:kill, Cint, (Cint, Cint), pid, 0)
    ret == 0 && return true
    return Libc.errno() == 1                # EPERM (1 on Linux and macOS) → exists, not ours
end

# Parse the embedded master pid out of a `jl_mlutils_shm_<pid>_<counter>_<rand>` dir name.
function _shm_dir_pid(name::AbstractString)
    parts = split(name, '_')
    length(parts) >= 4 || return nothing
    return tryparse(Int, parts[4])
end

# Remove session dirs left behind by *dead* master processes (crashed runs). Skips dirs whose
# embedded pid is still alive or not ours (EPERM), mirroring PyTorch's shared-memory reaper.
# Caller holds `_SHM_LOCK`.
function _shm_sweep_stale()
    root = _shm_root()
    isdir(root) || return nothing
    for name in readdir(root)
        startswith(name, "jl_mlutils_shm_") || continue
        pid = _shm_dir_pid(name)
        (pid === nothing || _pid_alive(pid)) && continue
        try
            rm(joinpath(root, name); recursive = true, force = true)
        catch
        end
    end
    return nothing
end

# Register the process-exit backstop once. `atexit` runs on a normal task, so it may use the
# recursive `rm` in `_shm_close_all`.
function _shm_register_atexit()
    _SHM_ATEXIT_REGISTERED[] && return nothing
    _SHM_ATEXIT_REGISTERED[] = true
    atexit(_shm_close_all)
    return nothing
end

# Create a fresh session for `owner` (the loader's `_cache` `RefValue`): sweep stale dirs, prune
# already-removed registry entries, make a private `0o700` dir, register it, and attach a
# finalizer to `owner` so the dir is torn down when the loader is garbage-collected. Returns
# `nothing` on an unsupported platform or if the dir cannot be created — the caller then degrades
# to the serialized path.
function _shm_session_create(owner)
    _shm_supported() || return nothing
    root = _shm_root()
    return lock(_SHM_LOCK) do
        _shm_register_atexit()
        _shm_sweep_stale()
        filter!(isdir, _SHM_SESSIONS)       # drop entries whose dir a finalizer already removed
        for _ in 1:8                        # retry the (astronomically unlikely) name collision
            counter = (_SHM_COUNTER[] += 1)
            name = string("jl_mlutils_shm_", Int(getpid()), "_", counter, "_", randstring(8))
            dir = joinpath(root, name)
            try
                mkdir(dir; mode = 0o700)
            catch
                continue
            end
            push!(_SHM_SESSIONS, dir)
            # The finalizer captures only `dir` (a String) — no back-reference to `owner`, so no
            # cycle keeps the loader alive. It does the minimal *non-yielding* teardown (see
            # `_shm_session_close`); recursive cleanup lives in `_shm_close_all`/`atexit`.
            finalizer(_ -> _shm_session_close(dir), owner)
            return ShmSession(dir, _SHM_MIN_BYTES[])
        end
        return nothing
    end
end

# Finalizer-safe teardown of a single session dir. Runs in finalizer context, where yielding I/O
# (e.g. the `scandir` inside a recursive `rm`) can throw "cannot wait on a task from a finalizer".
# In steady state the per-leaf unlink at unpack has already emptied the dir, so a plain `rm`
# (a synchronous `rmdir` ccall, no `readdir`) succeeds; if leaves are still in flight it fails
# with `ENOTEMPTY` and the `_shm_close_all`/`atexit` backstops (normal tasks) reclaim it later.
function _shm_session_close(dir::String)
    try
        rm(dir)                             # plain rmdir — no recursive readdir walk
    catch
    end
    return nothing
end

# Remove every registered session dir (recursively) and clear the registry. Called from
# `close_dataloader_pool` and `atexit` — both normal-task contexts, so the recursive `rm` is safe.
function _shm_close_all()
    lock(_SHM_LOCK) do
        for dir in _SHM_SESSIONS
            try
                rm(dir; recursive = true, force = true)
            catch
            end
        end
        empty!(_SHM_SESSIONS)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Pack (worker side): batch -> skeleton with `ShmHandle` leaves
# ---------------------------------------------------------------------------

# Number of bytes an array leaf would occupy densely (independent of any view/stride overhead).
_shm_nbytes(x::AbstractArray) = length(x) * sizeof(eltype(x))

# Is `x` a large, densely-serializable array leaf worth moving through shared memory? Ranges are
# O(1)-serializable (shipping the bytes would *inflate* them); offset axes cannot round-trip
# through a plain `Array`; non-bits eltypes cannot be memory-mapped.
function _shm_packable(x::AbstractArray, minbytes::Int)
    isbitstype(eltype(x)) || return false
    x isa AbstractRange && return false
    Base.has_offset_axes(x) && return false
    return _shm_nbytes(x) >= minbytes
end

# Public entry: pack a batch for transport, or a no-op when there is no session (`shm=false`).
shm_pack(x, ::Nothing) = x
shm_pack(x, session::ShmSession) = _shm_pack(x, session, _SHM_MAX_DEPTH)

# Anything not matched below (numbers, strings, ranges, opaque structs) passes through unchanged
# and is serialized as today — correct, just not accelerated. Also the depth backstop.
_shm_pack(x, ::ShmSession, ::Int) = x

# Multi-dimensional arrays: materialize dense into shared memory when large enough. A non-`Array`
# (SubArray/PermutedDimsArray/Adjoint/BitArray) is normalized to a dense `Array` on the receiver.
function _shm_pack(x::AbstractArray, session::ShmSession, depth::Int)
    return _shm_packable(x, session.minbytes) ? _shm_write_leaf(x, session) : x
end

# Vectors: a bits-eltype vector is a leaf; a non-bits vector (ragged / vector-of-observations) is
# packed element-wise.
function _shm_pack(x::AbstractVector, session::ShmSession, depth::Int)
    if isbitstype(eltype(x))
        return _shm_packable(x, session.minbytes) ? _shm_write_leaf(x, session) : x
    end
    depth <= 0 && return x
    items = Vector{Any}(undef, length(x))
    j = 0
    for el in x
        items[j += 1] = _shm_pack(el, session, depth - 1)
    end
    return ShmVector{eltype(x)}(items)
end

_shm_pack(x::Tuple, session::ShmSession, depth::Int) =
    depth <= 0 ? x : map(el -> _shm_pack(el, session, depth - 1), x)

_shm_pack(x::NamedTuple, session::ShmSession, depth::Int) =
    depth <= 0 ? x : map(el -> _shm_pack(el, session, depth - 1), x)

function _shm_pack(x::AbstractDict{K,V}, session::ShmSession, depth::Int) where {K,V}
    depth <= 0 && return x
    pairs = Vector{Pair{K,Any}}(undef, length(x))
    j = 0
    for (k, v) in x
        pairs[j += 1] = k => _shm_pack(v, session, depth - 1)
    end
    return ShmDict{K,V}(pairs)
end

# Write one array leaf into a fresh, collision-free file in the session dir and return its handle.
# `mktemp` gives a kernel-atomic unique name, so concurrent workers writing into the shared dir
# (every batch, every epoch) never alias. On any failure (dir gone after teardown, `ENOSPC`, …)
# clean up the partial file, warn once, and return the array inline — a per-leaf fallback that
# never hard-fails a batch mid-epoch.
function _shm_write_leaf(x::AbstractArray{T,N}, session::ShmSession) where {T,N}
    path = io = nothing
    try
        path, io = mktemp(session.dir; cleanup = false)
    catch e
        @warn "MLUtils shm: could not create a leaf file; falling back to serialization for this leaf" exception = (e, catch_backtrace()) maxlog = 1
        return x
    end
    try
        A = Mmap.mmap(io, Array{T,N}, size(x); shared = true)
        copyto!(A, x)
        # No `msync`: same-host readers map the same inode and see writes through the shared page
        # cache without it (msync only forces write-back to disk, irrelevant here and on tmpfs).
        finalize(A)                         # deterministic worker-side munmap
        close(io)
        return ShmHandle{T,N}(path, size(x))
    catch e
        try; close(io); catch; end
        try; rm(path; force = true); catch; end
        @warn "MLUtils shm: leaf write failed; falling back to serialization for this leaf" exception = (e, catch_backtrace()) maxlog = 1
        return x
    end
end

# ---------------------------------------------------------------------------
# Unpack (main side): skeleton with `ShmHandle` leaves -> batch of `Array`s
# ---------------------------------------------------------------------------

# Public entry: reconstruct a received skeleton, or a no-op when there is no session.
shm_unpack(x, ::Nothing) = x
shm_unpack(x, ::ShmSession) = _shm_unpack(x)

_shm_unpack(x) = x
_shm_unpack(h::ShmHandle) = _shm_read_leaf(h)
_shm_unpack(x::Tuple) = map(_shm_unpack, x)
_shm_unpack(x::NamedTuple) = map(_shm_unpack, x)

function _shm_unpack(x::ShmVector{E}) where {E}
    items = map(_shm_unpack, x.items)
    return all(el -> el isa E, items) ? convert(Vector{E}, items) : items
end

function _shm_unpack(d::ShmDict{K,V}) where {K,V}
    pairs = [k => _shm_unpack(v) for (k, v) in d.pairs]
    return all(p -> p.second isa V, pairs) ? Dict{K,V}(pairs) : Dict{K,Any}(pairs)
end

# Map a handle back to a writable `Array` and unlink the backing file immediately. The file is
# opened `"r+"` and mapped from the `IO` so the mapping is `PROT_READ|PROT_WRITE` — mutating a
# received batch (which training loops do) would segfault on a read-only mapping. `grow = false`
# makes a truncated/corrupt file fail loudly rather than silently zero-extend. After `rm`, POSIX
# keeps the pages valid until the returned array's finalizer munmaps it, so user-held batches
# stay valid indefinitely.
function _shm_read_leaf(h::ShmHandle{T,N}) where {T,N}
    io = open(h.file, "r+")
    try
        A = Mmap.mmap(io, Array{T,N}, h.dims; grow = false, shared = true)
        close(io)
        rm(h.file; force = true)
        return A
    catch e
        try; close(io); catch; end
        rethrow(e)
    end
end
