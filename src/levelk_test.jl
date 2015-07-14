using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps




coll = ColladaObjects("../data/maps/demo_map_border2D.dae");
map = UrbanMap(coll, 10, 10);

pomdp = SniperPOMDP(map)

n = pomdp.x_size * pomdp.y_size
b = zeros(n) + 1.0 / n 

#pomdpx = MOMDPX("test.pomdpx", initial_belief=b)
pomdpx = MOMDPX("../data/pomdpx/test.pomdpx")

write(pomdp, pomdpx)

