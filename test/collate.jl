

@testset "collate" begin
    # tuple and vector
    @test collate([(1, 2), (3, 4)]) == ([1, 3], [2, 4])
    # tuple, vector and arrays
    @test collate([([1, 2], 3), ([4, 5], 6)]) == ([1 4; 2 5], [3, 6])
    # named tuple, tuple and vector
    @test collate([(x = [1, 2], y = 3), (x = [4, 5], y = 6)]) ==
          (x = [1 4; 2 5], y = [3, 6])
    # dict and vector
    @test collate([Dict("x" => [1, 2], "y" => 3), Dict("x" => [4, 5], "y" => 6)]) ==
          Dict("x" => [1 4; 2 5], "y" => [3, 6])
end
