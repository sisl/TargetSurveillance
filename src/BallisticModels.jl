module BallisticModels


export
    BallisticModel,
    # Models
    Constant,
    SimplePoly,
    # methods
    prob

abstract BallisticModel

# Constant model (all prob is 1.0)
type Constant <: BallisticModel

end

# Polynomial model
type SimplePoly <: BallisticModel
    c::Vector{Float64} # poly coefficients ordered from lowest to highest order
    p1t::Vector{Float64} # pre-allocated for discrete inputs
    p2t::Vector{Float64}
    n::Int64 # length of poly
end
function SimplePoly(c::Vector{Float64})
    p1 = zeros(2)
    p2 = zeros(2)
    return SimplePoly(c, p1, p2, length(c))
end

prob(m::Constant, p1::Vector{Int64}, p2::Vector{Int64}, xs::Int64, ys::Int64) = 1.0
prob(m::Constant, p1::Vector{Float64}, p2::Vector{Float64}) = 1.0

# TODO: xs and ys should go in the BallisticModel type
# evaluate the hit probability for 2D discrete input
function prob(m::SimplePoly, p1::Vector{Int64}, p2::Vector{Int64}, xs::Int64, ys::Int64)
    p1t = m.p1t
    p2t = m.p2t
    for i = 1:length(p1)
        p1t[i] = p1[i] / xs
        p2t[i] = p2[i] / ys
    end
    return prob(m, p1t, p2t)
end

# evaluate the hit probability for 2D continuous input
function prob(m::SimplePoly, p1::Vector{Float64}, p2::Vector{Float64})
    d = sqrt( (p1[1] - p2[1])^2 + (p1[2] - p2[2])^2 )
    return prob(m, d)
end
function prob(m::SimplePoly, x::Float64)
    p = 0.0
    c = m.c
    for i = 1:m.n
        p += c[i] * x^(i-1)
    end
    return p
end

end #module


