@testset "eachobsparallel" begin

    @testset "Iteration" begin
        iter = eachobsparallel(collect(1:10))
        @test_nowarn for i in iter end
        X_ = collect(iter)
        @test all(x ∈ 1:10 for x in X_)
        @test length(unique(X_)) == 10
    end

    @testset "With `ThreadedEx`" begin
        iter = eachobsparallel(collect(1:10); executor = ThreadedEx())
        @test_nowarn for i in iter end
        X_ = collect(iter)
        @test all(x ∈ 1:10 for x in X_)
        @test length(unique(X_)) == 10
    end
end


@testset "RingBuffer" begin
    @testset "fills correctly" begin
        x = rand(10, 10)
        ringbuffer = RingBuffer([x, deepcopy(x), deepcopy(x)])
        @async begin
            for i ∈ 1:100
                put!(ringbuffer) do buf
                    rand!(buf)
                end
            end
        end
        @test_nowarn for _ ∈ 1:100
            take!(ringbuffer)
        end
    end

    @testset "does mutate" begin
        x = rand(10, 10)
        ringbuffer = RingBuffer([x, deepcopy(x), deepcopy(x)])
        put!(ringbuffer) do buf
            @test x ≈ buf
            copy!(buf, rand(10, 10))
            buf
        end
        x_ = take!(ringbuffer)
        @test !(x ≈ x_)
    end
end

@testset "`DataLoader(buffer = true, parallel=true)`" begin
    iter = DataLoader(collect(1:10), buffer=true, batchsize=-1, parallel=true)
    @test_nowarn for i in iter end
    X_ = collect(iter)
    @test all(x ∈ 1:10 for x in X_)
    @test length(unique(X_)) == 10
end
