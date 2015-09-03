using Colladas
using UrbanMaps
using MultiAgentSimulators
using NestedPolicies
using DiscreteSniper
using RolloutPolicies
using POMDPXFile


function load_names(map_type::Symbol, xs::Int64, ys::Int64, mu::Float64)
    map = None; target = None; name = None
    if map_type == :demo
        # demo
        coll = ColladaObjects("../data/maps/demo_map_border2D.dae");                            
        map = UrbanMap(coll, xs, ys);                                                           
        target = (0.2,0.8)
        name = "../data/policies/demo_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"              
    elseif map_type == :square
        # open square
        coll = ColladaObjects("../data/maps/square_map_2D_new.dae");
        map = UrbanMap(coll, xs, ys, shift=true, shift_val=0.5);
        target = (0.5, 0.5)
        name = "../data/policies/square_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"   
    elseif map_type == :rectangle
        # rectangle
        map = UrbanMap(1, [0.5], [0.5], [0.42], [0.22], xs, ys)
        target = (0.2,0.8)
        name = "../data/policies/rectangle_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"   
    elseif map_type == :pareto
        # open square
        coll = ColladaObjects("../data/maps/square_map_2D_new.dae");
        map = UrbanMap(coll, xs, ys, shift=true, shift_val=0.5);
        target = (0.5, 0.5)
        name = "../data/policies/square_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"   
    end
    return map, target, name
end


xs = 15
ys = 15
mu = 0.3                                                                                                                     
k = 2
mtype = :demo

map, target, pname = load_names(mtype, xs, ys, mu)

agents = [:sniper, :resource]

policy = NestedPolicy(pname)

p = None
pomdp = None

for agent in agents
    for i = 0:k
        println("Writing level-$i")
        name = "../data/pomdpx/$(string(mtype))_map/nested-$(string(agent))-$(xs)x$(ys)-mu-$(mu)-level-$i.pomdpx"
        pomdpx = MOMDPX(name)
        if i == 0
            pomdp = SniperPOMDP(map, adversary_prob=mu, target=target, agent=agent) 
        else
            if agent == :resource
                p = policy.adv_policies[i].policy
                pomdp = SniperPOMDP(map, adversary_prob=mu, target=target, agent=agent, adversary_policy=p, lvlk=true) 
            else
                p = policy.own_policies[i].policy
                pomdp = SniperPOMDP(map, adversary_prob=mu, target=target, agent=agent, adversary_policy=p, lvlk=true) 
            end
        end
        write(pomdp, pomdpx)
    end
end

