@testset "make_sin" begin
    n = 50
    xtmp, ytmp = make_sin(n; noise = 0.)

    @test length(xtmp) == length(ytmp) == n
    for i = 1:length(xtmp)
        @test sin.(xtmp[i]) ≈ ytmp[i]
    end
    # print(scatterplot(xtmp, ytmp; color = :blue, height = 5))
end

@testset "make_poly" begin
    coef = [.8, .5, 2]
    xtmp, ytmp = make_poly(coef, -10:.1:10; noise = 0)

    @test length(xtmp) == length(ytmp)
    for i = 1:length(xtmp)
        @test (coef[1] * xtmp[i]^2 + coef[2] * xtmp[i]^1 + coef[3]) ≈ ytmp[i]
    end
    # print(scatterplot(xtmp, ytmp; color = :blue, height = 5))
end

@testset "make_spiral" begin
    n = 97
    xtmp, ytmp = make_spiral(n; noise = 0.)

    @test length(xtmp[1, :]) == length(ytmp) == 2*n
    # test_plot = scatterplot(xtmp[1, 1:97], xtmp[2, 1:97], title="Spiral Function", color=:blue, name="pos")
    # print(scatterplot!(test_plot, xtmp[1, 98:194], xtmp[2, 98:194], color=:yellow, name="neg" ))
end


@testset "make_moons" begin
    x, y = Datasets.make_moons(100, noise=0, shuffle=false)
    @test size(x) == (2, 100)
    @test size(y) == (100,)
    @test all(==(1), y[1:50])
    @test all(==(2), y[51:100])
    @test minimum(x[1,1:50]) >= -1
    @test maximum(x[1,1:50]) <= 1
    @test minimum(x[2,1:50]) >= -1
    @test maximum(x[2,1:50]) <= 1
    @test minimum(x[1,51:100]) >= 0
    @test maximum(x[1,51:100]) <= 2
    @test minimum(x[2,51:100]) >= -0.5
    @test maximum(x[2,51:100]) <= 0.5
end