using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP
using NestedPolicies


function solve_nested(pomdp::SniperPOMDP, name::ASCIIString, k::Int64;
                      n_iter::Int64=25)
    policy = NestedPolicy(name)
    solver = NestedSolver(k = k, n_iter = n_iter)
    solve!(policy, solver, pomdp)
    write(policy)
end

function nested_pomdpx_file_name(pomdp::SniperPOMDP, k::Int64)
    mu = pomdp.adversary_prob
    xs = pomdp.x_size
    ys = pomdp.y_size
    as = ""
    agent == :resource ? (as="resource") : (as="sniper")
    name = "nested-$(as)-$(xs)x$(ys)-mu-$(mu)-level-$(k).pomdpx" 
    return name
end

# writes a pomdpx file and solves it?
function write_pomdpx(path::ASCIIString, pomdp::SniperPOMDP, k::Int64)
    file = nested_pomdpx_file_name(pomdp, k)
    pomdpx = MOMDPX("$(path)/$(file)")
    write(pomdp, pomdpx)
end

function valid_belief(pomdp::SniperPOMDP)
    return nothing 
end

xs = 15
ys = 15
mu = 0.3
k = 2
n_iter = 1

# rectangular map
#map = UrbanMap(1, [0.5], [0.5], [0.42], [0.22], xs, ys)
#name = "../data/policies/rectangle_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"
#name = "../data/policies/rectangle_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics-heuristic.nested"
#target = (0.2, 0.8)

# demo map
#coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
#map = UrbanMap(coll, xs, ys);
#target = (0.2, 0.8)
#name = "../data/policies/demo_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"
#name = "../data/policies/demo_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics-heuristic.nested"

# open square map
coll = ColladaObjects("../data/maps/square_map_2D_new.dae");
#map = UrbanMap(coll, xs, ys, shift=true, shift_val=0.5);
map = UrbanMap(coll, xs, ys);
target = (0.5, 0.5)
name = "../data/policies/square_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics-heuristic.nested"

# open square map thin walls
#coll = ColladaObjects("../data/maps/square_thin_map_2D.dae");
#map = UrbanMap(coll, xs, ys);
#target = (0.5, 0.5)
#name = "../data/policies/square_thin_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballistics.nested"
#name = "../data/policies/square_thin_map/$(xs)x$(ys)-mu-$(mu)-level-$(k)-ballisticsi-heuristic.nested"


#target_r = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
#target_r = [0.5, 0.75, 1.0]

#for r in target_r
#    name = "../data/policies/square_map/pareto/$(xs)x$(ys)-mu-$(mu)-level-$(k)-r-$(r)-ballistics-heuristic.nested"
#    policy = NestedPolicy(name)
#    solver = NestedSolver(k=k, n_iter=n_iter)
#    pomdp = SniperPOMDP(map, adversary_prob = mu, target = target, target_reward=r)
#    solve!(policy, solver, pomdp, dump=true)
#end

policy = NestedPolicy(name)
solver = NestedSolver(k=k, n_iter=n_iter)
pomdp = SniperPOMDP(map, adversary_prob = mu, target = target)

solve!(policy, solver, pomdp, dump=true)
#write(policy)

#pomdpx = MOMDPX("../data/pomdpx/demo_map/nested-resource-$(xs)x$(ys)-mu-$(mu)-level-2.pomdpx")
#write(pomdp, pomdpx)
