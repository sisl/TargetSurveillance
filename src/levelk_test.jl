using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP




coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
map = UrbanMap(coll, 10, 10);
pomdp = SniperPOMDP(map)


#pomdpx = MOMDPX("test.pomdpx", initial_belief=b)
#pomdpx = MOMDPX("../data/pomdpx/test.pomdpx")
#write(pomdp, pomdpx)

filename = "/Users/megorov/Desktop/projects/stanford/publications/conferences/icaart/2016/code/data/policies/resource-1.policy"

#alphas = MOMDPAlphas(filename)
policy = PolicyFile(filename)
ns = n_states(pomdp)
b = DiscreteBelief(ns)

