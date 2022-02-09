@testset "unsqueeze" begin
    x = randn(2, 3, 2)
    @test @inferred(unsqueeze(x; dims=1)) == reshape(x, 1, 2, 3, 2)
    @test @inferred(unsqueeze(x; dims=2)) == reshape(x, 2, 1, 3, 2)
    @test @inferred(unsqueeze(x; dims=3)) == reshape(x, 2, 3, 1, 2)
    @test @inferred(unsqueeze(x; dims=4)) == reshape(x, 2, 3, 2, 1)

    @test unsqueeze(dims=2)(x) == unsqueeze(x, dims=2)
end

@testset "stack and unstack" begin
    x = randn(3,3)
    stacked = stack([x, x], dims=2)
    @test size(stacked) == (3,2,3)

    stacked_array=[ 8 9 3 5; 9 6 6 9; 9 1 7 2; 7 4 10 6 ]
    unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
    @test unstack(stacked_array, dims=2) == unstacked_array
    @test stack(unstacked_array, dims=2) == stacked_array
    @test stack(unstack(stacked_array, dims=1), dims=1) == stacked_array
end

@testset "batch and unbatch" begin
    stacked_array=[ 8 9 3 5
                    9 6 6 9
                    9 1 7 2
                    7 4 10 6 ]
    unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
    @test unbatch(stacked_array) == unstacked_array
    @test batch(unstacked_array) == stacked_array

    # no-op for vector of non-arrays
    @test batch([1,2,3]) == [1,2,3]
    @test unbatch([1,2,3]) == [1,2,3]

    # generic iterable
    @test batch(ones(2) for i=1:3) == ones(2, 3)
    @test unbatch(ones(2, 3)) == [ones(2) for i=1:3]

    @testset "tuple" begin
        @test batch([(1,2), (3,4)]) == ([1,3], [2,4])
        @test batch([([1,2], [3,4]), ([5,6],[7,8])]) == ([1 5
                                                          2 6], 
                                                         [3 7
                                                          4 8])      
    end

    @testset "named tuple" begin
        @test batch([(a=1,b=2), (a=3,b=4)]) == (a=[1,3], b=[2,4])
        @test batch([(a=1,b=2), (b=4,a=3)]) == (a=[1,3], b=[2,4])
        nt = [(a=[1,2], b=[3,4]), (a=[5,6],b=[7,8])]
        @test batch(nt) == (a = [1 5
                                2 6], 
                            b = [3 7
                                4 8])      
    end

    @testset "dict" begin
        @test batch([Dict(:a=>1,:b=>2), Dict(:a=>3,:b=>4)]) == Dict(:a=>[1,3], :b=>[2,4])
        @test batch([Dict(:a=>1,:b=>2), Dict(:b=>4,:a=>3)]) == Dict(:a=>[1,3], :b=>[2,4])
        d = [Dict(:a=>[1,2], :b=>[3,4]), Dict(:a=>[5,6],:b=>[7,8])]
        @test batch(d) == Dict(:a => [1 5
                                      2 6], 
                                :b => [3 7
                                      4 8])    
    end
end

@testset "flatten" begin
    x = randn(Float32, 10, 10, 3, 2)
    @test size(flatten(x)) == (300, 2)
end

@testset "normalise" begin
    x = randn(Float32, 3, 2, 2)
    @test normalise(x) == normalise(x; dims=3)
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
    cs[1] == [1  6; 2  7; 3  8; 4  9; 5 10]
    cs[2] == [11 16; 12 17; 13 18; 14 19; 15 20]
    
    test_zygote(chunk, rand(10), 3)
end

@testset "group_counts" begin
    d = group_counts(['a','b','b'])
    @test d == Dict('a' => 1, 'b' => 2)
end

@testset "rpad" begin
    @test rpad([1, 2], 4, 0) == [1, 2, 0, 0]
    @test rpad([1, 2, 3], 2, 0) == [1,2,3]
end

@testset "batchseq" begin
    bs = batchseq([[1, 2, 3], [4, 5]], 0)
    @test bs[1] == [1, 4]
    @test bs[2] == [2, 5]
    @test bs[3] == [3, 0]

    bs = batchseq([[1, 2, 3], [4, 5]], -1)
    @test bs[1] == [1, 4]
    @test bs[2] == [2, 5]
    @test bs[3] == [3, -1]
end
