module RolloutPolicies

using MOMDPs
using DiscreteSniper

export
    RandomPolicy,
    action


type RandomPolicy
    pomdp::SniperPOMDP
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


end # module
