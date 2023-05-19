using MLUtils
struct DummyData{X}
    x::X
end
MLUtils.numobs(data::DummyData) = numobs(data.x)
MLUtils.getobs(data::DummyData, idx) = getobs(data.x, idx)
MLUtils.getobs!(buffer, data::DummyData, idx) = getobs!(buffer, data.x, idx)

data = DummyData(rand(3,100))
d1 = collect(DataLoader(data; batchsize=1, buffer=true)) # no error
d2 = collect(DataLoader(data; batchsize=-1, buffer=true)) # error
