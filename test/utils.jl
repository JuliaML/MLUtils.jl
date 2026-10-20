@testset "unsqueeze" begin
    x = randn(2, 3, 2)
    @test @inferred(unsqueeze(x; dims=1)) == reshape(x, 1, 2, 3, 2)
    @test @inferred(unsqueeze(x; dims=2)) == reshape(x, 2, 1, 3, 2)
    @test @inferred(unsqueeze(x; dims=3)) == reshape(x, 2, 3, 1, 2)
    @test @inferred(unsqueeze(x; dims=4)) == reshape(x, 2, 3, 2, 1)

    @test unsqueeze(dims=2)(x) == unsqueeze(x, dims=2)

    @test_throws AssertionError unsqueeze(rand(2,2), dims=4)

    # test gradient
    for d in (1, 2, 3, 4)
        test_zygote(unsqueeze, x, fkwargs=(; dims=d), check_inferred=false)
    end
end

@testset "stack and unstack" begin
    x = randn(3,3)
    stacked = stack([x, x], dims=2)
    @test size(stacked) == (3,2,3)
    @test @inferred(stack([x, x], dims=2)) == stacked

    stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
    unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
    @test unstack(stacked_array, dims=2) == unstacked_array
    @test @inferred(unstack(stacked_array, dims=Val(2))) == unstacked_array
    @test stack(unstacked_array, dims=2) == stacked_array
    @test stack(unstack(stacked_array, dims=1), dims=1) == stacked_array

    for d in (1,2,3)
        test_zygote(stack, [x,2x], fkwargs=(; dims=d), check_inferred=false)
    end

    # test gradient of unstack
    for d in (1,2)
        test_zygote(unstack, x, fkwargs=(; dims=d), check_inferred=false)
    end

    # Issue #121
    a = [[1] for i in 1:10000]
    @test size(stack(a, dims=1)) == (10000, 1)
    @test size(stack(a, dims=2)) == (1, 10000)
end

@testset "flatten" begin
    x = randn(Float32, 10, 10, 3, 2)
    @test size(flatten(x)) == (300, 2)
end

@testset "normalise" begin
    x = randn(Float32, 3, 2, 2)
    @test normalise(x) == normalise(x; dims=3)

    x = randn(Float32, 3, 4)
    y = normalise(x; dims=1, ϵ=0)
    @test mean(y; dims=1) ≈ zeros(Float32, 1, 4) atol=1e-5
    @test std(y; dims=1, corrected=false) ≈ ones(Float32, 1, 4) atol=1e-5
end

@testset "rescale" begin
    x = randn(Float32, 3, 2, 2)
    @test rescale(x) == rescale(x; dims=3)

    x = randn(Float32, 3, 4)
    y = rescale(x; dims=1, ϵ=0)
    @test minimum(y; dims=1) ≈ zeros(Float32, 1, 4) atol=1e-5
    @test maximum(y; dims=1) ≈ ones(Float32, 1, 4) atol=1e-5
    @test all(0 .<= y .<= 1)
end

@testset "chunk" begin
    cs = chunk(collect(1:10), 3)
    @test length(cs) == 3
    @test cs[1] == [1, 2, 3, 4]
    @test cs[2] == [5, 6, 7, 8]
    @test cs[3] == [9, 10]

    cs = chunk(collect(1:10), 3)
    @test length(cs) == 3
    @test cs[1] == [1, 2, 3, 4]
    @test cs[2] == [5, 6, 7, 8]
    @test cs[3] == [9, 10]
    
    x = reshape(collect(1:20), (5, 4))
    cs = chunk(x, 2)
    @test length(cs) == 2
    @test cs[1] == [1  6; 2  7; 3  8; 4  9; 5 10]
    @test cs[2] == [11 16; 12 17; 13 18; 14 19; 15 20]

    x = permutedims(reshape(collect(1:10), (2, 5)))
    cs = chunk(x; size = 2, dims = 1)
    @test length(cs) == 3
    @test cs[1] == [1 2; 3 4]
    @test cs[2] == [5 6; 7 8]
    @test cs[3] == [9 10]

    # test gradient
    test_zygote(chunk, rand(10), 3, check_inferred=false)

    # indirect test of second order derivates
    n = 2
    dims = 2
    x = rand(4, 5)
    l = chunk(x, 2)
    dl = randn!.(collect.(l))
    idxs = MLUtils._partition_idxs(x, cld(size(x, dims), n), dims)
    # `project`/`xproto` stand in for `x` in `∇chunk`; they are non-differentiable,
    # so hold them fixed while checking the gradient wrt `dl`.
    project = ProjectTo(x)
    xproto = similar(x, 0)
    test_zygote(MLUtils.∇chunk, dl, project ⊢ NoTangent(), xproto ⊢ NoTangent(), idxs, Val(dims), check_inferred=false)


    if CUDA.functional()
        # https://github.com/JuliaML/MLUtils.jl/issues/103
        x = rand(2, 10) |> cu
        cs = chunk(x, 2)
        @test length(cs) == 2
        @test cs[1] isa CuArray
        @test cs[1] == x[:, 1:5]
    end

    @testset "size collection" begin
        a = reshape(collect(1:10), (5, 2))
        y = chunk(a; dims = 1, size = (1, 4))
        @test length(y) == 2
        @test y[1] == [1 6]
        @test y[2] == [2 7; 3 8; 4 9; 5 10]

        test_zygote(x -> chunk(x; dims = 1, size = (1, 4)), a, check_inferred=false)
    end

    @testset "chunk by partition_idxs" begin
        x = reshape(collect(1:15), (3, 5))
        partition_idxs = [1,1,3,3,4]

        y = chunk(x, partition_idxs)
        @test length(y) == 4
        @test y[1] == [1 4; 2 5; 3 6]
        @test size(y[2]) == (3, 0)
        @test y[3] == [7 10; 8 11; 9 12]
        @test y[4] == reshape([13, 14, 15], 3, 1)

        y = chunk(x, partition_idxs; npartitions=5)
        @test length(y) == 5
        @test size(y[5]) == (3, 0)

        y = chunk(x, [1,1,2]; dims=1)
        @test length(y) == 2
        @test y[1] == [1 4 7 10 13; 2 5 8 11 14]
        @test y[2] == [3 6 9 12 15]

        @test chunk(x, [1,1,2]; dims=Val(1)) == y
    end

    @testset "type stability (#260)" begin
        x = rand(Float32, 8, 3)
        @test @inferred(chunk(x, 4; dims = Val(1))) == chunk(x, 4; dims = 1)
        @test @inferred(chunk(x; size = 2, dims = Val(2))) == chunk(x; size = 2, dims = 2)
        @test @inferred(chunk(x; size = [1, 2], dims = Val(2))) == chunk(x; size = [1, 2], dims = 2)

        # a constant `dims` is propagated through the keyword call
        f1 = x -> chunk(x, 4; dims = 1)
        f2 = x -> chunk(x, 4)
        f3 = x -> chunk(x; size = 2, dims = 1)
        @test @inferred(f1(x)) == chunk(x, 4; dims = 1)
        @test @inferred(f2(x)) == chunk(x, 4)
        @test @inferred(f3(x)) == chunk(x; size = 2, dims = 1)

        x = rand(8, 3)
        y, back = @inferred MLUtils.rrule(chunk, x; size = 2, dims = Val(1))
        @test y == chunk(x; size = 2, dims = 1)
        @inferred back(collect.(y))
        test_zygote(x -> chunk(x, 4; dims = Val(1)), x, check_inferred=false)
        test_zygote(x -> chunk(x; size = [1, 2], dims = Val(2)), x, check_inferred=false)
    end
end

@testset "group_counts" begin
    d = group_counts(['a','b','b'])
    @test d == Dict('a' => 1, 'b' => 2)
end

@testset "ones_like" begin
    x = rand(Float16, 2, 3)
    y = ones_like(x, (2, 4, 2))
    @test y isa AbstractArray{Float16}
    @test y == ones(Float16, 2, 4, 2)

    test_zygote(ones_like, rand(5), (2, 4, 2))
end

@testset "zeros_like" begin
    x = rand(Float16, 2, 3)
    y = zeros_like(x, (2, 4, 2))
    @test y isa AbstractArray{Float16}
    @test y == zeros(Float16, 2, 4, 2)

    test_zygote(zeros_like, rand(5), (2, 4, 2))
end

@testset "rand_like" begin
    seed = 42
    x = rand(Float16, 2, 3)
    y = rand_like(rr(seed), x, (2, 4, 2))
    @test y isa AbstractArray{Float16}
    @test y == rand(rr(seed), Float16, 2, 4, 2)
end

@testset "randn_like" begin
    seed = 42
    x = rand(Float16, 2, 3)
    y = randn_like(rr(seed), x, (2, 4, 2))
    @test y isa AbstractArray{Float16}
    @test y == randn(rr(seed), Float16, 2, 4, 2)
end

@testset "fill_like" begin
    x = rand(Float16, 2, 3)
    y = fill_like(x, 2.2, (2, 4, 2))
    @test y isa AbstractArray{Float16}
    @test y == fill!(rand(Float16, 2, 4, 2), 2.2)

    test_zygote(fill_like, rand(5), rand(), (2, 4, 2))
end

@testset "trues_like and falses_like" begin
    x = rand(Float16, 2, 3)
    y = trues_like(x, (2, 4, 2))
    @test y isa Array{Bool,3}
    @test y == trues(2, 4, 2)

    y = falses_like(x, (2, 4, 2))
    @test y isa Array{Bool,3}
    @test y == falses(2, 4, 2)
end

@testset "rpad_constant" begin
    @test rpad_constant([1, 2], 4, -1) == [1, 2, -1, -1]
    @test rpad_constant([1, 2, 3], 2)  == [1, 2, 3]
    @test rpad_constant([1 2; 3 4], 4; dims=1) == [1 2; 3 4; 0 0; 0 0]
    @test rpad_constant([1 2; 3 4], 4) == [1 2 0 0; 3 4 0 0; 0 0 0 0; 0 0 0 0]
    @test rpad_constant([1 2; 3 4], (3, 4)) == [1 2 0 0; 3 4 0 0; 0 0 0 0]
end

@testset "batched_searchsorted" begin
    # vector haystack matches Base.searchsorted{first,last} elementwise
    v = sort(randn(20))
    q = randn(8)
    @test batched_searchsortedfirst(v, q) == searchsortedfirst.(Ref(v), q)
    @test batched_searchsortedlast(v, q) == searchsortedlast.(Ref(v), q)

    # descending haystack
    vr = sort(randn(20); rev=true)
    @test batched_searchsortedfirst(vr, q; rev=true) == searchsortedfirst.(Ref(vr), q; rev=true)
    @test batched_searchsortedlast(vr, q; rev=true) == searchsortedlast.(Ref(vr), q; rev=true)

    # documented examples
    @test batched_searchsortedfirst([1, 3, 5, 7, 9], [4, 5, 8]) == [3, 3, 5]
    @test batched_searchsortedlast([1, 3, 5, 7, 9], [4, 5, 8]) == [2, 3, 4]

    # batched along dim 1 (last dim is the batch): each column searched independently
    xp = sort(randn(10, 5); dims=1)
    x = randn(4, 5)
    rf = batched_searchsortedfirst(xp, x)
    rl = batched_searchsortedlast(xp, x)
    @test size(rf) == size(x) == size(rl)
    for j in 1:5
        @test rf[:, j] == searchsortedfirst.(Ref(xp[:, j]), x[:, j])
        @test rl[:, j] == searchsortedlast.(Ref(xp[:, j]), x[:, j])
    end

    # batched along a non-default dim
    xp2 = sort(randn(5, 10); dims=2)
    x2 = randn(5, 4)
    r2 = batched_searchsortedlast(xp2, x2; dims=2)
    @test size(r2) == size(x2)
    for i in 1:5
        @test r2[i, :] == searchsortedlast.(Ref(xp2[i, :]), x2[i, :])
    end

    # a vector haystack broadcasts against a batched query
    @test batched_searchsortedfirst(v, x) == searchsortedfirst.(Ref(v), x)

    if CUDA.functional()
        CUDA.allowscalar(false)
        try
            xpg, xg = cu(xp), cu(x)
            for f in (batched_searchsortedfirst, batched_searchsortedlast)
                rg = f(xpg, xg)
                @test rg isa CuArray
                @test Array(rg) == f(xp, x)
            end
        finally
            CUDA.allowscalar(true)
        end
    end
end

@testset "topk" begin
    # vector
    @test topk([5, 1, 4, 2, 3], 3) == ([5, 4, 3], [1, 3, 5])
    @test topk([5, 1, 4, 2, 3], 2; rev=false) == ([1, 2], [2, 4])

    # the returned indices index back into the selected dimension
    x = randn(6, 8)
    for dims in (1, 2), rev in (true, false)
        n = size(x, dims)
        vals, inds = topk(x, 3; dims, rev)
        @test size(vals) == size(inds)
        sorted = sort(x; dims, rev)
        @test vals == selectdim(sorted, dims, 1:3)
        for I in CartesianIndices(inds)
            J = Base.setindex(Tuple(I), inds[I], dims)
            @test x[J...] == vals[I]
        end
    end

    topk_dims(x, dims) = topk(x, 2; dims)
    @test @inferred(topk_dims(x, 2)) isa Tuple{Matrix{Float64}, Matrix{Int}}

    # argument checking
    @test_throws ArgumentError topk([1, 2, 3], 0)
    @test_throws ArgumentError topk([1, 2, 3], 4)

    # gradient flows through the values only, scattered back to the selected entries
    for dims in (1, 2)
        x = randn(Float32, 5, 4)
        g = Zygote.gradient(z -> sum(abs2, first(topk(z, 2; dims))), x)[1]
        vals, inds = topk(x, 2; dims)
        expected = zero(x)
        for I in CartesianIndices(inds)
            J = Base.setindex(Tuple(I), inds[I], dims)
            expected[J...] = 2 * x[J...]
        end
        @test g ≈ expected
    end

    if CUDA.functional()
        # forbid scalar indexing to make sure topk stays on the GPU
        CUDA.allowscalar(false)
        try
            x = randn(Float32, 6, 8)
            xg = cu(x)
            for dims in (1, 2), rev in (true, false)
                vc, ic = topk(x, 3; dims, rev)
                vg, ig = topk(xg, 3; dims, rev)
                @test vg isa CuArray
                @test ig isa CuArray
                @test Array(vg) ≈ vc
                @test Array(ig) == ic
            end
            gg = Zygote.gradient(z -> sum(abs2, first(topk(z, 3; dims=1))), xg)[1]
            gc = Zygote.gradient(z -> sum(abs2, first(topk(z, 3; dims=1))), x)[1]
            @test gg isa CuArray
            @test Array(gg) ≈ gc
        finally
            CUDA.allowscalar(true)
        end
    end
end
@testset "sample_logits" begin
    # deterministic cases
    logits = [1.0 5.0 0.0; 3.0 0.0 2.0; 2.0 1.0 9.0]
    @test sample_logits(logits; temperature=0) == [2 1 3]
    @test sample_logits(logits; k=1) == [2 1 3]
    @test sample_logits(logits; p=1e-3) == [2 1 3]
    @test sample_logits(permutedims(logits); dims=2, temperature=0) == reshape([2, 1, 3], 3, 1)
    @test sample_logits([1.0, 3.0, 2.0]; temperature=0) == [2]
    @test sample_logits([1.0, 3.0, 2.0]; k=1, p=0.5) == [2]
    @test sample_categorical([0.0 0.0; 1.0 0.0; 0.0 3.0]) == [2 3]
    @test sample_logits(logits; n=2, temperature=0) == [2 1 3; 2 1 3]
    @test sample_logits(logits; n=2, temperature=0, replace=false) == [2 1 3; 3 3 2]
    @test sample_categorical([0.0 0.0; 1.0 0.0; 0.0 3.0]; n=3) == [2 3; 2 3; 2 3]

    # the output has the shape of argmax(x; dims), with size n along dims
    x = randn(Float32, 4, 5, 6)
    for dims in 1:3
        idx = sample_logits(x; dims, k=2, p=0.9)
        @test idx isa Array{Int,3}
        @test size(idx) == size(argmax(x; dims))
        @test all(1 .<= idx .<= size(x, dims))
        for replace in (false, true)
            idx = sample_logits(x; dims, n=3, replace, k=3)
            @test idx isa Array{Int,3}
            @test size(idx) == Base.setindex(size(x), 3, dims)
            @test all(1 .<= idx .<= size(x, dims))
        end
    end
    @test size(sample_logits(logits; n=5)) == (5, 3)
    @test all(c -> sort(c) == 1:10, eachcol(sample_logits(randn(10, 1000); n=10, replace=false)))
    sample_dims(x, dims) = sample_logits(x; dims, n=2, k=3, p=0.9)
    @test @inferred(sample_dims(x, 2)) isa Array{Int,3}
    @test sample_logits(Xoshiro(1), x; k=3) == sample_logits(Xoshiro(1), x; k=3)
    @test sample_categorical(Xoshiro(1), abs.(x)) == sample_categorical(Xoshiro(1), abs.(x))

    # the frequencies match the target distribution
    softmax(z) = (e = exp.(z .- maximum(z)); e ./ sum(e))
    freqs(idx) = [count(==(i), idx) / length(idx) for i in 1:6]
    l = [2.0, 1.0, 0.5, -1.0, 0.0, 1.5]   # softmax(l) ≈ [0.42, 0.15, 0.09, 0.02, 0.06, 0.25]
    top2 = softmax(ifelse.(l .>= 1.5, l, -Inf))
    top3 = softmax(ifelse.(l .>= 1.0, l, -Inf))
    L = repeat(l, 1, 100_000)
    rng = Xoshiro(0)
    @test freqs(sample_logits(rng, L)) ≈ softmax(l) atol=0.01
    @test freqs(sample_logits(rng, permutedims(L); dims=2)) ≈ softmax(l) atol=0.01
    @test freqs(sample_logits(rng, L; temperature=0.5)) ≈ softmax(2l) atol=0.01
    @test freqs(sample_logits(rng, L; k=3)) ≈ top3 atol=0.01
    @test freqs(sample_logits(rng, L; k=3, temperature=2)) ≈ softmax(ifelse.(l .>= 1.0, l / 2, -Inf)) atol=0.01
    @test freqs(sample_logits(rng, L; p=0.7)) ≈ top3 atol=0.01
    # top-p applies to the top-k distribution: p=0.72 keeps 2 of the top 4, but 3 of all 6
    @test freqs(sample_logits(rng, L; k=4, p=0.72)) ≈ top2 atol=0.01
    @test freqs(sample_logits(rng, Float16.(L); k=3)) ≈ top3 atol=0.01
    @test all(in((1, 2, 6)), sample_logits(rng, L; k=3))
    @test freqs(sample_categorical(rng, L .- minimum(l))) ≈ (l .- minimum(l)) ./ 10 atol=0.01

    # joint frequencies of the first two of n draws
    q = softmax(l)
    joint(idx) = [count(c -> c[1] == i && c[2] == j, eachcol(idx)) / size(idx, 2) for i in 1:6, j in 1:6]
    joint_norep = [i == j ? 0.0 : q[i] * q[j] / (1 - q[i]) for i in 1:6, j in 1:6]
    @test joint(sample_logits(rng, L; n=2)) ≈ q * q' atol=0.01
    @test joint(sample_logits(rng, L; n=2, replace=false)) ≈ joint_norep atol=0.01
    @test joint(sample_logits(rng, L; n=3, replace=false)) ≈ joint_norep atol=0.01
    @test joint(permutedims(sample_logits(rng, permutedims(L); dims=2, n=2, replace=false))) ≈ joint_norep atol=0.01
    @test all(c -> sort(c) == [1, 6], eachcol(sample_logits(rng, L; n=2, k=2, replace=false)))
    @test freqs(vec(sample_categorical(rng, L .- minimum(l); n=2))) ≈ (l .- minimum(l)) ./ 10 atol=0.01

    @test_throws ArgumentError sample_logits(logits; k=0)
    @test_throws ArgumentError sample_logits(logits; p=0)
    @test_throws ArgumentError sample_logits(logits; p=1.5)
    @test_throws ArgumentError sample_logits(logits; temperature=-1)
    @test_throws ArgumentError sample_logits(logits; n=0)
    @test_throws ArgumentError sample_logits(logits; n=4, replace=false)

    @test Zygote.gradient(z -> sum(sample_logits(z)), randn(3, 4)) == (nothing,)

    if CUDA.functional()
        CUDA.allowscalar(false)
        try
            Lg = cu(L)
            for idx in (sample_logits(Lg; k=4, p=0.72),
                        sample_logits(CUDA.default_rng(), Lg; k=4, p=0.72),
                        permutedims(sample_logits(permutedims(Lg); dims=2, k=4, p=0.72)))
                @test idx isa CuArray{Int}
                @test freqs(Array(idx)) ≈ top2 atol=0.01
            end
            @test Array(sample_logits(cu(logits); temperature=0)) == [2 1 3]
            @test Array(sample_logits(cu(logits); n=2, temperature=0)) == [2 1 3; 2 1 3]
            @test Array(sample_logits(cu(logits); n=2, temperature=0, replace=false)) == [2 1 3; 3 3 2]
            idx = sample_logits(Lg; n=2, replace=false)
            @test idx isa CuArray{Int}
            @test joint(Array(idx)) ≈ joint_norep atol=0.01
            idx = sample_logits(Lg; n=2)
            @test idx isa CuArray{Int}
            @test joint(Array(idx)) ≈ q * q' atol=0.01
            idx = sample_categorical(cu(L .- minimum(l)))
            @test idx isa CuArray{Int}
            @test freqs(Array(idx)) ≈ (l .- minimum(l)) ./ 10 atol=0.01
        finally
            CUDA.allowscalar(true)
        end
    end
end
