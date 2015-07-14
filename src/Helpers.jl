module Helpers

export ind2sub!

# not exported in julia base?
function ind2sub!{T<:Integer}(sub::Array{T}, dims::Array{T}, ind::T)
    ndims = length(dims)
    stride = dims[1]
    for i in 2:(ndims - 1)
        stride *= dims[i]
    end
    for i in (ndims - 1):-1:1
        rest = rem1(ind, stride)
        sub[i + 1] = div(ind - rest, stride) + 1
        ind = rest
        stride = div(stride, dims[i])
    end
    sub[1] = ind
    return
end


end # module
