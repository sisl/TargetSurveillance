using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP
using POMDPToolbox
using MOMDPs
using GameServers


sn = int(ARGS[1])

coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
map = UrbanMap(coll, 10, 10);
pomdp = SniperPOMDP(map)

filename = "/Users/megorov/Desktop/projects/stanford/publications/conferences/icaart/2016/code/data/policies/resource-1.policy"

policy = PolicyFile(filename, :momdp)

server = SniperServer(sn)
start(server, pomdp, policy)
