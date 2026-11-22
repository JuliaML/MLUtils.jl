@setup_workload begin
    x = rand(Float32, 3, 8)
    y = rand(Float32, 1, 8)
    labels = rand(1:2, 8)
    xs = [rand(Float32, 3) for _ in 1:4]
    @compile_workload begin
        for data in (x, (x, y), (x, labels), (; x, y))
            for _ in DataLoader(data; batchsize = 4, shuffle = true) end
            for _ in eachobs(data) end
            splitobs(data; at = 0.5, shuffle = true)
            show(devnull, MIME"text/plain"(), DataLoader(data))
        end
        batch(xs)
        unbatch(x)
        stack(xs)
        unstack(x; dims = 2)
        flatten(x)
        unsqueeze(x; dims = 1)
        chunk(x, 2)
    end
end
