@testset "mapobs" begin
    data = 1:10
    mdata = mapobs(-, data)
    @test getobs(mdata, 8) == -8

    mdata2 = mapobs((-, x -> 2x), data)
    @test getobs(mdata2, 8) == (-8, 16)

    nameddata = mapobs((x = sqrt, y = log), data)
    @test getobs(nameddata, 10) == (x = sqrt(10), y = log(10))
    @test getobs(nameddata.x, 10) == sqrt(10)
end

@testset "filterobs" begin
    data = 1:10
    fdata = filterobs(>(5), data)
    @test numobs(fdata) == 5
end

@testset "groupobs" begin
    data = -10:10
    datas = groupobs(>(0), data)
    @test length(datas) == 2
end

@testset "joinobs" begin
    data1, data2 = 1:10, 11:20
    jdata = joinobs(data1, data2)
    @test getobs(jdata, 15) == 15
end
