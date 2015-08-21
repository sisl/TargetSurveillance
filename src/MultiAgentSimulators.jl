module MultiAgentSimulators

using DiscreteSniper
using MOMDPs
using RayCasters

export
    TwoAgetnSim,
    TwoMDPSim,
    TwoPOMDPSim,
    TwoMixedSim,
    batch,
    simulate!,
    reset!,
    stats


abstract TwoAgetnSim

# MDP vs MDP policy
type TwoMDPSim <: TwoAgetnSim
    nsteps::Int64 # number of times steps in sim
    s1::Int64 # initial state 1
    s2::Int64 # initial state 2
    r::Float64 # total reward
    shot::Int64 # num times resource shot
    obs::Int64 # num times target observed
    moves::Int64 # total number of moves
end
function TwoMDPSim(s1::Int64, s2::Int64, n::Int64)
    return TwoMDPSim(n, s1, s2, 0.0, 0, 0, 0)
end

# POMDP vs POMDP policy
type TwoPOMDPSim <: TwoAgetnSim
    nsteps::Int64 # number of times steps in sim
    s1::Int64 # initial state 1
    s2::Int64 # initial state 2
    b1::Belief # inital belief 1
    b2::Belief # initial belief 2
    r::Float64 # total reward
    shot::Int64 # num times resource shot
    obs::Int64 # num times target observed
    moves::Int64 # total number of moves
end

# MDP vs POMDP policy
type TwoMixedSim <: TwoAgetnSim

end

stats(sim::TwoAgetnSim) = (sim.r, sim.shot, sim.obs, sim.moves)

function reset!(sim::TwoMDPSim, s1::Int64, s2::Int64)
    sim.s1 = s1
    sim.s2 = s2
    sim.r = 0.0
    sim.shot = sim.obs = sim.moves = 0
    sim
end

function reset!(sim::TwoMDPSim, s1::Int64, s2::Int64, b1::Belief, b2::Belief)
    sim.s1 = s1
    sim.s2 = s2
    sim.b1 = b1
    sim.b2 = b2
    sim.r = 0.0
    sim.shot = sim.obs = sim.moves = 0
    sim
end

function simulate!(sim::TwoMDPSim, 
                  pomdp::SniperPOMDP,
                  p1::Policy,
                  p2::Policy;
                  verbose::Bool=true)

    map = pomdp.map
    s1i = sim.s1
    s2i = sim.s2
    s1agg = aggrogate(pomdp, s1i, s2i) 
    s2agg = aggrogate(pomdp, s2i, s1i) 
    s1 = deepcopy(s1i)
    s2 = deepcopy(s2i)
    target = pomdp.target
    pos1 = [1,1]; pos2 = [1,1]
    i2p!(pos1, pomdp, s1)
    i2p!(pos2, pomdp, s2)
    r = 0.0; shot = 0; obs = 0; moves = 0
    for i = 1:sim.nsteps
        verbose ? println("State: $s1agg, $s2agg, $pos1, $pos2, Stats: $r, $shot, $obs, $moves") : nothing
        a1 = action(p1, s1agg)
        a2 = action(p2, s2agg)
        # count statistics 
        r += reward(pomdp, s1agg, a1)
        isVisible(map, pos1, pos2) ? (shot+=1) : (nothing) 
        isVisible(map, pos1, target) ? (obs+=1) : (nothing)
        a1 != 1 ? (moves+=1) : (nothing)
        # move each entity
        move!(pos1, pomdp, s1, a1)
        move!(pos2, pomdp, s2, a2)
        s1 = p2i(pomdp, pos1)
        s2 = p2i(pomdp, pos2)
        s1agg = aggrogate(pomdp, s1, s2) 
        s2agg = aggrogate(pomdp, s2, s1) 
    end
    sim.r=r; sim.shot=shot; sim.obs=obs; sim.moves=moves
    sim
end



end # module
