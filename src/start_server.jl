using POMDPXFile
using DiscreteSniper
using Colladas
using UrbanMaps
using SARSOP
using POMDPToolbox
using MOMDPs
using GameServers
using NestedPolicies


# input args
sn = int(ARGS[1]) # socket number
mfile = ARGS[2] # .dae map file
ppath = ARGS[3] # path to the policy folder
agent = ARGS[4] # agent type (sniper or resource)
msize = int(ARGS[5]) # size of the map
mu = float(ARGS[6]) # stochasticity constant
k = int(ARGS[7]) # nested policy k
kp = int(ARGS[8]) # POMDP policy k


#################################################################
##################### EXAMPLE SCRIPT ############################
#################################################################
# julia start_server.jl 4444 ../data/maps/square_map_2D_new.dae ../data/policies/square_map resource 10 0.3 2 0 

ai = 0
agent == "resource" ? (ai = 1) : (ai = 2)

xs = msize
ys = msize

#mfile = "../data/maps/demo_map_border2D.dae"
mfile = convert(ASCIIString, mfile)
@assert isfile(mfile) "Invalid map file"
coll = ColladaObjects(mfile);
map = UrbanMap(coll, xs, ys);

pfile = "$(xs)x$(ys)-mu-$mu-level-$k-ballistics.nested"
pfile = convert(ASCIIString, joinpath(ppath, pfile))

@assert isfile(pfile) "Invalid MDP policy file"
mdp_policy = NestedPolicy(pfile)
p = get_policy(mdp_policy, kp, ai).policy
pomdp = SniperPOMDP(map, adversary_policy=p, adversary_prob=mu, lvlk=true, agent=:resource)

ppfile = "nested-$agent-$(xs)x$(ys)-mu-$(mu)-level-$(kp).policy"
ppfile = joinpath(ppath, ppfile) 

@assert isfile(ppfile) "Invalid POMDP policy file"
policy = PolicyFile(ppfile, :momdp)

# This needs to be zero if map is not square
shift = 0.5/xs
server = SniperServer(sn, shift=shift)
start(server, pomdp, policy)
