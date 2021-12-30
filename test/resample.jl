@testset "oversample" begin
    x = rand(2, 5)
    ya = ["a", "c", "c", "a", "b"]
    y2 = ["c", "c", "c", "a", "b"]
    
    o = oversample((x, ya), fraction=1, shuffle=false) 
    @test o == oversample((x, ya), ya, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 6)
    @test size(oy) == (6,)
    @test ox[:,1:5] == x
    @test ox[:,6] == ox[:,5]
    @test oy[1:5] == ya
    @test oy[6] == ya[5]

    o = oversample((x, ya), y2, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 9)
    @test size(oy) == (9,)
    @test ox[:,1:5] == x
    @test ox[:,6] == ox[:,7] == x[:,5]
    @test ox[:,8] == ox[:,9] == x[:,4]
    @test oy[1:5] == ya
    @test oy[6] == oy[7] == ya[5]
    @test oy[8] == oy[9] == ya[4]
end

@testset "undersample" begin
    x = rand(2, 5)
    ya = ["a", "c", "c", "a", "b"]
    y2 = ["c", "c", "c", "a", "b"]
    
    o = undersample((x, ya), shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 3)
    @test size(oy) == (3,)
    @test ox[:,3] == x[:,5]
    
    o = undersample((x, ya), y2, shuffle=false) 
    ox, oy = getobs(o)
    @test ox isa Matrix
    @test oy isa Vector 
    @test size(ox) == (2, 3)
    @test size(oy) == (3,)
    @test ox[:,2:3] == x[:,4:5]
end
