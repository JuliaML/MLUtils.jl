@testset "oversample" begin
    x = rand(2, 5)
    y = ["a", "c", "c", "a", "b"]
    y2 = ["c", "c", "c", "a", "b"]
    
    o = oversample((x, y), fraction=1, shuffle=false) 
    @test o == oversample((x, y), y, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 6)
    @test size(oy) == (6,)
    @test ox[:,1:5] == x
    @test ox[:,6] == ox[:,5]
    @test oy[1:5] == y
    @test oy[6] == y[5]

    o = oversample((x, y), y2, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 9)
    @test size(oy) == (9,)
    @test ox[:,1:5] == x
    @test ox[:,6] == ox[:,7] == x[:,5]
    @test ox[:,8] == ox[:,9] == x[:,4]
    @test oy[1:5] == y
    @test oy[6] == oy[7] == y[5]
    @test oy[8] == oy[9] == y[4]
end

@testset "undersample" begin
    x = rand(2, 5)
    y = ["a", "c", "c", "a", "b"]
    y2 = ["c", "c", "c", "a", "b"]
    
    o = undersample((x, y), shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 3)
    @test size(oy) == (3,)
    @test oy == ["a", "c", "b"]
    @test ox[:,3] == x[:,5]
    
    o = undersample((x, y), y2, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 3)
    @test size(oy) == (3,)
    @test oy == ["a", "a", "b"]
    @test ox[:,2:3] == x[:,4:5]
end
