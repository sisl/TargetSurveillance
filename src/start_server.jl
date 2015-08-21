using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP
using POMDPToolbox
using MOMDPs
using GameServers
using NestedPolicies


sn = int(ARGS[1])
mfile = ARGS[2]
ppath = ARGS[3]
agent = ARGS[4]
msize = int(ARGS[5])
mu = float(ARGS[6])
k = int(ARGS[7])
kp = int(ARGS[8])

ai = 0
agent == "resource" ? (ai = 1) : (ai = 2)

xs = msize
ys = msize

mfile = "../data/maps/demo_map_border2D.dae"
@assert isfile(mfile) "Invalid map file"
coll = ColladaObjects(mfile);
map = UrbanMap(coll, xs, ys);

pfile = "$(xs)x$(ys)-mu-$mu-level-$k.nested"
pfile = convert(ASCIIString, joinpath(ppath, pfile))

@assert isfile(pfile) "Invalid policy file"
mdp_policy = NestedPolicy(pfile)
p = get_policy(mdp_policy, k, ai).policy
pomdp = SniperPOMDP(map, adversary_policy=p, adversary_prob=mu, lvlk=true, agent=:resource)

#filename = "../data/policies/demo_map/resource-1-20x20.policy"
ppfile = "nested-$agent-$(xs)x$(ys)-mu-$(mu)-level-$(kp).policy"
ppfile = joinpath(ppath, ppfile) 

policy = PolicyFile(ppfile, :momdp)

server = SniperServer(sn)
start(server, pomdp, policy)
