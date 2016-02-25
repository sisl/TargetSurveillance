module Heuristics

using DiscreteValueIteration
using POMDPs
using DiscreteSniper

export
    LocalSearch,
    action,
    value


type LocalSearch <: Policy
    vipolicy::ValueIterationPolicy
    pomdp::SniperPOMDP
end
function LocalSearch(pomdp::POMDP, q::Matrix{Float64})
    vip = DiscreteValueIteration(pomdp, q)
    return LocalSearch(vip, pomdp)
end


POMDPs.action(policy::LocalSearch, s::Int64) = action(policy.vipolicy, s)
POMDPs.value(policy::LocalSearch, s::Int64) = value(policy.vipolicy, s)

function POMDPs.action(policy::LocalSearch, b::Belief)
    s = indmax(b)
    return action(policy.vipolicy, s)
end
function POMDPs.value(policy::LocalSearch, b::Belief)
    s = indmax(b)
    return value(policy.vipolicy, s)
end

function POMDPs.action(policy::LocalSearch, b::Belief, x::Int64)
    y = indmax(b.b)
    s = xy2i(policy.pomdp, [x,y])
    return action(policy.vipolicy, s)
end

function POMDPs.value(policy::LocalSearch, b::Belief, x::Int64)
    y = indmax(b.b)
    s = xy2i(policy.pomdp, [x,y])
    return value(policy.vipolicy, s)
end
end # module
