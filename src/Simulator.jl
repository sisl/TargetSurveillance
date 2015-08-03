module Simulator

using RayCasters
using DiscreteSniper
using POMDPToolbox


export simulate, simulateStep!
export Simulation, MDPSimulation, POMDPSimulation


abstract Simulation

type MDPSimulation <: Simulation
    coordinates::Matrix{Float64}
    rewards::Vector{Float64}
    time::Vector{Int64}
    actions::Vector{Int64}
    nObs::Int64
    nKills::Int64
    nMoves::Int64
    lastState::Int64
end
function MDPSimulation(mdp::POMDP, init_state::Int64, n_steps::Int64;
                       n_vars::Int64=6)
    coords = zeros(nSteps, n_vars)
    t      = zeros(Int64, n_steps)
    a      = zeros(Int64, n_steps)
    r      = zeros(n_steps)
    obs    = 0
    kills  = 0
    moves  = 0
    return MDPSimulation(coords, r, t, a, obs, kills, moves, init_state)
end


type POMDPSimulation <: Simulation
    coordinates::Matrix{Float64}
    rewards::Vector{Float64}
    time::Vector{Int64}
    actions::Vector{Int64}
    nObs::Int64
    nKills::Int64
    nMoves::Int64
    belief::Matrix{Float64}
    lastState::Int64
    lastBelief::Vector{Float64}
end
function POMDPSimulation(pomdp::POMDP, init_state::Int64, n_steps::Int64;
                       n_vars::Int64=6, init_belief::Vector{Float64}=Float64[])
    n_s = n_states(pomdp)
    coords = zeros(n_steps, n_vars)
    belief = zeros(n_steps, n_s)
    t      = zeros(Int64, n_steps)
    a      = zeros(Int64, n_steps)
    r      = zeros(n_steps)
    obs    = 0
    kills  = 0
    moves  = 0
    return POMDPSimulation(coords, r, t, a, obs, kills, moves, belief, init_state, init_belief)
end


type MOMDPSimulation <: Simulation
    coordinates::Matrix{Float64}
    rewards::Vector{Float64}
    time::Vector{Int64}
    actions::Vector{Int64}
    no::Int64
    nk::Int64
    nm::Int64
    belief::Matrix{Float64}
    last_monitor::Int64 # fully observable var index
    last_threat::Int64 # part observable var index
    lastBelief::Vector{Float64} # belief over partially obs vars
end
function MOMDPSimulation(pomdp::POMDP, init_state::Int64, n_steps::Int64;
                       n_vars::Int64=6, init_belief::Vector{Float64}=Float64[])
    n_s = n_states(pomdp)
    coords = zeros(n_steps, n_vars)
    belief = zeros(n_steps, n_s)
    t      = zeros(Int64, n_steps)
    a      = zeros(Int64, n_steps)
    r      = zeros(n_steps)
    obs    = 0
    kills  = 0
    moves  = 0
    return POMDPSimulation(coords, r, t, a, obs, kills, moves, belief, init_state, init_belief)
end


# target and sniper start in same location
function simulate(policy::Policy, mdp::DiscreteMDP, nSteps::Int, startM::Array{Int64})
    sizes  = mdp.dimSizes 
    target = mdp.targets[1]
    startS = zeros(Int64, 2)
    startS[1] = target[1]
    startS[2] = target[2]
    startX = [startS, startM]
    s = sub2ind(sizes, startX)
    return simulate(policy, mdp, nSteps; initState=s)
end


function simulate(policy::Policy, mdp::DiscreteMDP, nSteps::Int; initState = 10)

    # check that init is valid state

    sim = MDPSimulation(mdp, initState, nSteps)
    for i = 1:nSteps
        simulateStep!(sim, mdp, i, policy)
    end
    return sim
end


function simulateStep!(sim::MDPSimulation, mdp::DiscreteMDP, step::Int, policy::Policy)
    # check if last state is valid state
    map = mdp.map
    s = sim.lastState
    a = action(policy, s)

    r     = 0.0
    obs   = sim.nObs
    kills = sim.nKills
    moves = sim.nMoves

    r = reward(mdp, s, a)
    obs += numTargetsObserved(mdp)
    kills += numMonitorsObserved(mdp)

    # generative model
    sp = nextState(mdp, s, a)

    i2x!(mdp, s) # fills tempSub1
    nextCoords = mdp.tempSub1
    targetPos = vcat([vcat(mdp.targets[i]...) for i = 1:length(mdp.targets)]...) # ugly conversion

    sim.coordinates[step,:] = [nextCoords, targetPos]
    sim.rewards[step]       = r
    sim.time[step]          = step
    sim.actions[step]       = a 

    sim.nObs      = obs
    sim.nKills    = kills
    sim.nMoves    = moves
    sim.lastState = sp
    return
end



function simulate!(sim::MOMDPSimulation, policy::Policy, pomdp::POMDP, nsteps::Int64)

    # check that init is valid state
    for i = 1:nSteps
        simulateStep!(sim, pomdp, policy, i)
    end
    return sim
end

function simulateStep!(sim::MOMDPSimulation, policy::Policy, pomdp::POMDP, step::Int64)

end


end # module
