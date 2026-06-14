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
    test_zygote(MLUtils.∇chunk, dl, x, idxs, Val(dims), check_inferred=false)


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

        test_zygote(x -> chunk(x; dims = 1, size = (1, 4)), a)
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