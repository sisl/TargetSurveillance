module MultiAgentSimulators

using DiscreteSniper
using MOMDPs
using RayCasters
using BallisticModels
using POMDPToolbox

export
    TwoAgentSim,
    TwoMDPSim,
    TwoPOMDPSim,
    TwoMixedSim,
    batch,
    simulate!,
    reset!,
    stats,
    full_stats


abstract TwoAgentSim

# MDP vs MDP policy
type TwoMDPSim <: TwoAgentSim
    nsteps::Int64 # number of times steps in sim
    s1::Int64 # initial state 1
    s2::Int64 # initial state 2
    ttk::Float64 # mean time to kill
    r::Float64 # total reward
    shot::Int64 # num times resource shot
    obs::Int64 # num times target observed
    moves::Int64 # total number of moves
    prf::Vector{Int64} # position where the resource was killed
    psf::Vector{Int64} # position where the sniper was when killed
end
function TwoMDPSim(s1::Int64, s2::Int64, n::Int64)
    return TwoMDPSim(n, s1, s2, 0.0, 0.0, 0, 0, 0, [-1,-1], [-1,-1])
end

# POMDP vs POMDP policy
type TwoPOMDPSim <: TwoAgentSim
    nsteps::Int64 # number of times steps in sim
    s1::Int64 # initial state 1
    s2::Int64 # initial state 2
    b1::Belief # inital belief 1
    b2::Belief # initial belief 2
    ttk::Int64 # time to kill
    r::Float64 # total reward
    shot::Int64 # num times resource shot
    obs::Int64 # num times target observed
    moves::Int64 # total number of moves
    prf::Vector{Int64} # position where the resource was killed
    psf::Vector{Int64} # position where the sniper was when killed
    n_part_obs_states::Int64
    obs_dist1::AbstractDistribution
    obs_dist2::AbstractDistribution
end
function TwoPOMDPSim(s1::Int64, s2::Int64, b1::Belief, b2::Belief, n::Int64)
    return TwoPOMDPSim(n, s1, s2, b1, b2, 0.0, 0.0, 0, 0, 0, [-1,-1], [-1,-1])
end
function TwoPOMDPSim(pomdp::SniperPOMDP, s1::Int64, s2::Int64, n::Int64)
    # belief uniform for adversary and initially known for own
    d1 = create_observation_distribution(pomdp)
    d2 = create_observation_distribution(pomdp)
    posize = length(collect(domain(part_obs_space(pomdp))))
    b1 = DiscreteBelief(posize)
    b2 = DiscreteBelief(posize)
    fill!(b1, 0.0)
    b1[s2] = 1.0
    fill!(b2, 1/posize)
    return TwoPOMDPSim(n, s1, s2, b1, b2, 0.0, 0.0, 0, 0, 0, [-1,-1], [-1,-1], posize, d1, d2)
end

# MDP vs POMDP policy
type TwoMixedSim <: TwoAgentSim
    nsteps::Int64 # number of times steps in sim
    s1::Int64 # initial state 1
    s2::Int64 # initial state 2
    b1::Belief # inital belief 1
    b2::Belief # initial belief 2
    ttk::Int64 # time to kill
    r::Float64 # total reward
    shot::Int64 # num times resource shot
    obs::Int64 # num times target observed
    moves::Int64 # total number of moves
    prf::Vector{Int64} # position where the resource was killed
    psf::Vector{Int64} # position where the sniper was when killed
    n_part_obs_states::Int64
    obs_dist1::AbstractDistribution
    obs_dist2::AbstractDistribution
end

stats(sim::TwoAgentSim) = (sim.r, sim.shot, sim.obs, sim.moves, sim.ttk)
full_stats(sim::TwoAgentSim) = (sim.r, sim.shot, sim.obs, sim.moves, sim.ttk, sim.prf, sim.psf)

function reset!(sim::TwoMDPSim, s1::Int64, s2::Int64)
    sim.s1 = s1
    sim.s2 = s2
    sim.r = 0.0
    sim.shot = sim.obs = sim.moves = sim.ttk = 0
    fill!(sim.prf, -1)
    fill!(sim.psf, -1)
    sim
end

function reset!(sim::TwoPOMDPSim, s1::Int64, s2::Int64)
    sim.s1 = s1
    sim.s2 = s2
    ns = sim.n_part_obs_states
    fill!(sim.b2, 1/ns)
    fill!(sim.b1, 0.0)
    sim.b1[s2] = 1.0
    sim.r = 0.0
    sim.shot = sim.obs = sim.moves = sim.ttk = 0
    fill!(sim.prf, -1)
    fill!(sim.psf, -1)
    sim
end

function simulate!(sim::TwoMDPSim, 
                  pomdp::SniperPOMDP,
                  p1::Policy,
                  p2::Policy;
                  verbose::Bool=true,
                  debug::Bool=false)
    
    ballistic_model = pomdp.ballistic_model
    map = pomdp.map
    xs = pomdp.x_size
    ys = pomdp.y_size
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
    r = 0.0; shot = 0; obs = 0; moves = 0; ttk = 0
    for i = 1:sim.nsteps
        verbose ? println("State: $s1agg, $s2agg, $pos1, $pos2, Stats: $r, $shot, $obs, $moves") : nothing
        a1 = action(p1, s1agg)
        a2 = action(p2, s2agg)
        # count statistics 
        currentr = reward(pomdp, s1agg, a1)
        ttk = i
        #r += reward(pomdp, s1agg, a1)
        sprob = prob(ballistic_model, pos1, pos2, xs, ys)
        rn = rand()
        if isVisible(map, pos1, pos2) && rn < sprob
            shot += 1
            r += currentr
            copy!(sim.prf, pos1)
            copy!(sim.psf, pos2)
            break
        end
        # only add sniper reward if shot
        currentr > -0.1 ? (r+=currentr) : (nothing)
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
    sim.r=r; sim.shot=shot; sim.obs=obs; sim.moves=moves; sim.ttk = ttk
    sim
end


# both p1 and p2 are MOMDP policies
function simulate!(sim::TwoPOMDPSim, 
                  pomdp::SniperPOMDP,
                  p1::Policy,
                  p2::Policy;
                  verbose::Bool=true,
                  debug::Bool=false)
    
    ballistic_model = pomdp.ballistic_model
    map = pomdp.map
    xs = pomdp.x_size
    ys = pomdp.y_size
    s1i = sim.s1
    s2i = sim.s2
    # get aggrogate state indices
    s1agg = aggrogate(pomdp, s1i, s2i) 
    s2agg = aggrogate(pomdp, s2i, s1i) 
    s1 = deepcopy(s1i)
    s2 = deepcopy(s2i)
    target = pomdp.target
    pos1 = [1,1]; pos2 = [1,1]
    # get true positions on the grid
    i2p!(pos1, pomdp, s1)
    i2p!(pos2, pomdp, s2)
    # initialize belief
    b1 = sim.b1 
    b2 = sim.b2 
    # pre-allocated distributions
    od1 = sim.obs_dist1
    od2 = sim.obs_dist2


    r = 0.0; shot = 0; obs = 0; moves = 0; ttk = 0

    for i = 1:sim.nsteps
        verbose ? println("State: $s1agg, $s2agg, $pos1, $pos2, Stats: $r, $shot, $obs, $moves") : nothing
        a1 = action(p1, b1, s1)
        a2 = action(p2, b2, s2)
        # count statistics 
        currentr = reward(pomdp, s1agg, a1)
        ttk = i
        sprob = prob(ballistic_model, pos1, pos2, xs, ys)
        rn = rand()
        # end sim if shot
        if isVisible(map, pos1, pos2) && rn < sprob
            shot += 1
            r += currentr
            copy!(sim.prf, pos1)
            copy!(sim.psf, pos2)
            break
        end
        # only add sniper reward if shot
        currentr > -0.1 ? (r+=currentr) : (nothing)
        isVisible(map, pos1, target) ? (obs+=1) : (nothing)
        a1 != 1 ? (moves+=1) : (nothing)
        # move each entity
        move!(pos1, pomdp, s1, a1)
        move!(pos2, pomdp, s2, a2)
        s1 = p2i(pomdp, pos1)
        s2 = p2i(pomdp, pos2)
        # get the observation
        observation!(od1, pomdp, s1, s2, a1)
        observation!(od2, pomdp, s2, s1, a2)
        o1 = rand(od1)
        o2 = rand(od2)
        debug ? println("\nResource: $s1, $a1, $o1, \n $(b1.b)\n") : nothing
        debug ? println("\nThreat: $s2, $a2, $o2, \n $(b2.b)\n") : nothing
        update_belief!(b1, pomdp, s1, a1, o1)
        update_belief!(b2, pomdp, s2, a2, o2)
        s1agg = aggrogate(pomdp, s1, s2) 
        s2agg = aggrogate(pomdp, s2, s1)
    end
    sim.r=r; sim.shot=shot; sim.obs=obs; sim.moves=moves; sim.ttk = ttk
    sim
end

end # module
