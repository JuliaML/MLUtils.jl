@testset "slidingwindow" begin
    data = 1:20
    s = slidingwindow(data, size=5)
    @test length(s) == 16
    @test s[1] == 1:5
    @test s[2] == 2:6

    s = slidingwindow(data, size=5, stride=3)
    @test length(s) == 6
    @test s[1] == 1:5
    @test s[2] == 4:8
    @test s[3] == 7:11
    @test s[4] == 10:14
    @test s[5] == 13:17
    @test s[6] == 16:20

    data = reshape(1:18, 3, 6)
    s = slidingwindow(data, size=2)
    @test length(s) == 5
    @test s[1] isa Matrix{Int}
    @test s[1] == [1 4; 2 5; 3 6]
    @test s[2] == [4 7; 5 8; 6 9]

    c = 0
    for w in s
        @test w isa Matrix{Int}
        @test size(w) == (3, 2)
        c += 1
    end
    @test c == 5

    @testset "obsdim" begin
        x = rand(2, 6, 4)
        v = slidingwindow(x, size=2, obsdim=2)
        @test length(v) == 5
        @test v[1] isa Array{Float64,3}
        @test v[1] == x[:, 1:2, :]
        @test v[2] == x[:, 2:3, :]
        @test v[3] == x[:, 3:4, :]
        @test v[4] == x[:, 4:5, :]
    end
end
