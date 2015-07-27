using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP
using POMDPToolbox
using MOMDPs
using GameServers



coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
map = UrbanMap(coll, 10, 10);
pomdp = SniperPOMDP(map)


#pomdpx = MOMDPX("test.pomdpx", initial_belief=b)
#pomdpx = MOMDPX("../data/pomdpx/test.pomdpx")
#write(pomdp, pomdpx)

filename = "/Users/megorov/Desktop/projects/stanford/publications/conferences/icaart/2016/code/data/policies/resource-1.policy"

#alphas = MOMDPAlphas(filename)
policy = PolicyFile(filename, :momdp)
ns = length(collect(domain(part_obs_space(pomdp))))
b = DiscreteBelief(ns)

server = SniperServer(4444)

start(server, pomdp, policy)
