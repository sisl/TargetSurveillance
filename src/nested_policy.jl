using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using DiscreteValueIteration


function make_nested_policy(map::UrbanMap, mu::Float64, k::Int64;
                            n_iter::Int64=50, eps::Float64=1e-3,
                            path::ASCIIString="../data/policies/",
                            saveq::Bool=true)
    # find level-0 policy
    pomdp = SniperPOMDP(map)
    policy = ValueIterationPolicy(pomdp)
    solver = ValueIterationSolver(n_iter, eps)
    solve!(policy, solver, pomdp, verbose=true)
    name = "$(path)nested-$(pomdp.x_size)x$(pomdp.y_size)-mu-$(mu)-level-0.Q" 
    saveq ? writedlm(name, policy.qmat) : nothing

    # find k = i level policies
    for i = 1:k
        println("\nStarting: Level-$i\n")
        p = policy.policy 
        pomdp = SniperPOMDP(map, adversary_policy=p, adversary_prob=mu, lvlk=true)
        policy = ValueIterationPolicy(pomdp)
        solve!(policy, solver, pomdp, verbose=true)
        name = "$(path)nested-$(pomdp.x_size)x$(pomdp.y_size)-mu-$(mu)-level-$(i).Q" 
        saveq ? writedlm(name, policy.qmat) : nothing
    end
    return policy
end


coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
map = UrbanMap(coll, 5, 5);

name = "../data/policies/nested-5x5-mu-0.6-level-2.policy"
mu = 0.6
k = 2

p = make_nested_policy(map, mu, k, filename=name)
