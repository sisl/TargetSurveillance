module RolloutPolicies

using MOMDPs
using DiscreteSniper

export
    RandomPolicy,
    SniperHeurisitcPolicy,
    action

import POMDPs: action


type RandomPolicy <: Policy
    pomdp::SniperPOMDP
end

# local search for point maximizing the metric: wt * dt + wr * dr
type SniperHeurisitcPolicy <: Policy
    pomdp::SniperPOMDP
    wt::Float64 # target weight
    wr::Float64 # resource weight
end
function SniperHeurisitcPolicy(pomdp::SniperPOMDP)
    return SniperHeurisitcPolicy(pomdp, 0.1, 1.0)
end

# returns a valid random action
function action(policy::RandomPolicy, s::Int64)
    pomdp = policy.pomdp
    xy = pomdp.temp_position
    n = n_actions(pomdp)
    i2xy!(xy, pomdp, s)
    p = xy[1]
    a = rand(1:n)
    while !valid_action(pomdp, p, a)
        a = rand(1:n)
    end
    return a
end

# returns a valid random action
function action(policy::RandomPolicy, s::Int64, b::Belief)
    pomdp = policy.pomdp
    xy = pomdp.temp_position
    n = n_actions(pomdp)
    i2xy!(xy, pomdp, s)
    p = xy[1]
    a = rand(1:n)
    while !valid_action(pomdp, p, a)
        a = rand(1:n)
    end
    return a
end

function action(policy::SniperHeurisitcPolicy, s::Int64)
    # find point with highest score
    # brute force enumerate
    
    # find unit vector direction
    # pick action that goes in that direction
end

end # module
