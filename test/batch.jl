
@testset "batch and unbatch" begin
    stacked_array=[ 8 9 3 5
                    9 6 6 9
                    9 1 7 2
                    7 4 10 6 ]
    unstacked_array=[[8, 9, 9, 7], [9, 6, 1, 4], [3, 6, 7, 10], [5, 9, 2, 6]]
    
    @test @inferred(unbatch(stacked_array)) == unstacked_array
    @test @inferred(batch(unstacked_array)) == stacked_array

    # no-op for vector of non-arrays
    @test batch([1,2,3]) == [1,2,3]
    @test unbatch([1,2,3]) == [1,2,3]

    # batching multi-dimensional arrays
    x = map(_ -> rand(4), zeros(2, 3))
    @test size(batch(x)) == (4, 2, 3)

    x = map(_ -> rand(4, 5), zeros(2, 3, 6))
    @test size(batch(x)) == (4, 5, 2, 3, 6)

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

@testset "batchseq" begin
    bs = batchseq([[1, 2, 3], [4, 5]], 0)
    @test bs[1] == [1, 4]
    @test bs[2] == [2, 5]
    @test bs[3] == [3, 0]

    bs = batchseq([[1, 2, 3], [4, 5]], -1)
    @test bs[1] == [1, 4]
    @test bs[2] == [2, 5]
    @test bs[3] == [3, -1]

    bs = batchseq([[ones(3), ones(3), ones(3)], [zeros(3), zeros(3)]], [-1,-1,-1])
    @test bs isa Vector{Matrix{Float64}}
    @test bs[1] == [1.0 0.0; 1.0 0.0; 1.0 0.0]
    @test bs[2] == [1.0 0.0; 1.0 0.0; 1.0 0.0]
    @test bs[3] ==  [1.0 -1.0; 1.0 -1.0; 1.0 -1.0]

    bs = batchseq([ones(2,4), zeros(2, 3), ones(2,2)])
    @test bs isa Vector{Matrix{Float64}}
    @test bs[1] == [1.0 0.0 1.0; 1.0 0.0 1.0]
    @test bs[2] == [1.0 0.0 1.0; 1.0 0.0 1.0]
    @test bs[3] == [1.0 0.0 0.0; 1.0 0.0 0.0]
    @test bs[4] == [1.0 0.0 0.0; 1.0 0.0 0.0]
end

@test "batch_sequences" begin
    y = batchseqs([[1, 2, 3], [10, 20]])
    @test y isa Matrix{Int}
    @test y == [1 10; 2 20; 3 0]

    data = (ones(2, 1), fill(2.0, (2, 3)))
    y = batchseqs(data, -1)
    @test y[:,:,1] == [1 -1 -1; 1 -1 -1]
    @test y[:,:,2] == [2 2 2; 2 2 2]

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = batch_sequences(x, 3, 2)
    @test y == [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10]]
end
