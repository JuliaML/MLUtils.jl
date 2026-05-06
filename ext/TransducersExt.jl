module TransducersExt

using MLUtils: DataLoader, numobs, getobs, getobs!, _shuffledata
import Transducers

@inline function _dataloader_foldl1(rf, val, d::DataLoader, data)
    if d.shuffle
        return _dataloader_foldl2(rf, val, d, _shuffledata(d.rng, data))
    else
        return _dataloader_foldl2(rf, val, d, data)
    end
end

@inline function _dataloader_foldl2(rf, val, d::DataLoader, data)
    if d.buffer == false
        return _dataloader_foldl3(rf, val, data)
    else
        return _dataloader_foldl3_buffered(rf, val, data, d.buffer)
    end
end

@inline function _dataloader_foldl3(rf, val, data)
    for i in 1:numobs(data)
        @inbounds x = getobs(data, i)
        # TODO: in 1.8 we could @inline this at the callsite,
        #       optimizer seems to be very sensitive to inlining and
        #       quite brittle in its capacity to keep this type stable
        val = Transducers.@next(rf, val, x)
    end
    return Transducers.complete(rf, val)
end

@inline function _dataloader_foldl3_buffered(rf, val, data, buf)
    for i in 1:numobs(data)
        @inbounds x = getobs!(buf, data, i)
        val = Transducers.@next(rf, val, x)
    end
    return Transducers.complete(rf, val)
end

@inline function Transducers.__foldl__(rf, val, d::DataLoader)
    d.parallel && throw(ArgumentError("Transducer fold protocol not supported on parallel data loads"))
    return _dataloader_foldl1(rf, val, d, d._data)
end

end # module
