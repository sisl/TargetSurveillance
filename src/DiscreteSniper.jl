module DiscreteSniper

export 
    SniperPOMDP,
    create_fully_obs_transition,
    create_partially_obs_transition,
    create_observation,
    weight, 
    index, 
    fully_obs_space,
    part_obs_space,
    actions,
    actions!,
    observations,
    observations!,
    n_states,
    n_actions,
    n_observations,
    domain,
    reward,
    transition!,
    observation!,
    move,
    move!,
    inbounds,
    i2p!,
    p2i

using MOMDPs
using POMDPToolbox
using Distributions

using Maps
using UrbanMaps
using Helpers # ind2sub!
using RayCasters # inBuilding


import MOMDPs: create_fully_obs_transition, create_partially_obs_transition, create_observation
import MOMDPs: weight, index, fully_obs_space, part_obs_space
import MOMDPs: n_states, n_actions, n_observations 
import MOMDPs: actions, actions!, observations, observations!, domain
import MOMDPs: reward, transition!, observation!



type SniperPOMDP <: MOMDP

    x_size::Int64
    y_size::Int64
    grid_size::Int64
    sizes::Vector{Int64}
    sub_sizes::(Int64,Int64)
    aggrogate_sizes::Vector{Int64}

    n_actions::Int64

    r_obs::Float64
    r_shot::Float64
    r_move::Float64

    adversary_model::Vector{Int64} # policy for adversary
    adversary_prob::Float64 # prob the adversary follows model
    invalid_positions::Set{Int64}

    temp_position::Vector{Int64}
    temp_position2::Vector{Int64}
    
    target::(Int64,Int64)

    action_map::Matrix{Float64}

    null_obs::Int64

    agent::Symbol
    lvlk::Bool

    map::Map

    # pre-allocated for memory efficiency

    function SniperPOMDP(map::Map; nSnipers::Int64 = 1, nMonitors::Int64 = 1,
                       target = (2,8),
                       target_reward = 0.1, sniper_reward = -1.0, move_reward = -0.01,
                       agent::Symbol = :resource, lvlk::Bool=false)
        self = new()

        x_size = map.xSize
        y_size = map.ySize

        self.x_size = x_size
        self.y_size = y_size
        self.grid_size = x_size*y_size
        self.sizes = [x_size, y_size]
        self.aggrogate_sizes = [x_size*y_size, x_size*y_size]
        self.sub_sizes = (x_size, y_size)

        self.null_obs = x_size*y_size+1

        self.r_obs = target_reward
        self.r_shot = sniper_reward
        self.r_move = move_reward

        self.target = target

        self.map = map

        self.temp_position = [1,1]
        self.temp_position2 = [1,1]
        self.invalid_positions = get_invalid_positions(map, x_size, y_size)

        action_map = [0 1 -1 0 0; 0 0 0 1 -1]
        self.action_map = action_map 
        self.n_actions = size(action_map, 2)

        self.agent = agent # :resource or :sniper
        self.lvlk = lvlk

        return self
    end
end

abstract SniperDistribution <: AbstractDistribution

# Fully Observable Distribution
type FODistribution <: SniperDistribution 
    interps::Interpolants
end

# Partially Observable Distribution
type PODistribution <: SniperDistribution
    interps::Interpolants
end

# currently interps are zero length
function create_fully_obs_transition(pomdp::SniperPOMDP)
    interps = Interpolants(1)
    push!(interps, 1, 1.0)
    d = FODistribution(interps)
    return d
end


function create_partially_obs_transition(pomdp::SniperPOMDP)
    na = n_actions(pomdp)
    interps = Interpolants(na)
    p = 1.0 / na
    for i = 1:na; push!(interps, i, p); end;
    PODistribution(interps)
end

weight(d::SniperDistribution, i::Int64) = d.interps.weights[i]
index(d::SniperDistribution, i::Int64) = d.interps.indices[i]
Base.length(d::SniperDistribution) = d.interps.length

n_states(pomdp::SniperPOMDP) = (pomdp.x_size * pomdp.y_size)^2
n_actions(pomdp::SniperPOMDP) = pomdp.n_actions
n_observations(pomdp::SniperPOMDP) = pomdp.x_size * pomdp.y_size + 1 


function transition!(d::FODistribution, pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    interps = d.interps
    invalids = pomdp.invalid_positions
    temp = pomdp.temp_position
    # if x is not a valid position then no movement
    if in(x, invalids)
        interps.indices[1] = x
    # otherwise motion is deterministic
    else
        move!(temp, pomdp, x, a)
        xp = p2i(pomdp, temp)
        # if bump into a building or go out, return to same state
        if in(xp, invalids) || !inbounds(pomdp.map, temp)
            interps.indices[1] = x
        else
            interps.indices[1] = xp
        end
    end
    d
end


function transition!(d::PODistribution, pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    if pomdp.lvlk
        lvlk_transition!(d, pomdp, x, y, a)
    else
        stochastic_transition!(d, pomdp, x, y, a)
    end
end

# stochastic
function stochastic_transition!(d::PODistribution, pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    interps = d.interps
    invalids = pomdp.invalid_positions
    if in(y, invalids)
        fill!(interps.weights, 0.0)
        interps.indices[1] = y
        interps.weights[1] = 1.0
    else
        temp = pomdp.temp_position
        na = n_actions(pomdp)
        fill!(interps.weights, 0.2) 
        inside = na
        for i = 1:na
            move!(temp, pomdp, y, i)
            yp = p2i(pomdp, temp)
            interps.indices[i] = yp
            if in(yp, pomdp.invalid_positions) || !inbounds(pomdp.map, temp)
                interps.weights[i] = 0.0
                interps.indices[i] = y
                inside -= 1
            end
        end
        # TODO: inside is an int, could impact performance
        new_prob = 1.0 / inside 
        for i = 1:na
            if interps.weights[i] > 0.0
                interps.weights[i] = new_prob
            end
        end
    end
    d
end

# level-k
# need the aggrogate state
function lvlk_transition!(d::PODistribution, pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    interps = d.interps
    na = n_actions(pomdp)
    fill!(interps.weights, 0.2) 
    ag = aggrogate(pomdp, x, y)
    oa = pomdp.adversary_model[ag]

    inside = 1.0

    for i = 1:na
        yp = move(pomdp, y, i)
        if in(yp, pomdp.invalid_positions)
            interps.weights[i] = 0.0
            inside += 1.0
        else
            interps.indices[i] = yp
        end
    end
    uni_prob = 1.0 / inside 
    for i = 1:na
        if interps.weights[i] > 0.0 && i != oa
            interps.weights[i] = uni_prob
        end
    end
    d
end

function move!(p::Vector{Int64}, pomdp::SniperPOMDP, x::Int64, a::Int64)
    sizes = pomdp.sizes
    am = pomdp.action_map 
    ind2sub!(p, sizes, x) 
    p[1] += am[1,a] 
    p[2] += am[2,a] 
    p
end



function aggrogate(pomdp::SniperPOMDP, x::Int64, y::Int64)
    as = pomdp.aggrogate_sizes # e.g [100,100] for 10x10 grid
    temp = pomdp.temp_position
    temp[1] = x
    temp[2] = y
    s = ind2sub(as, temp)
    return s
end


# Fully Observable Distribution
type ObsDistribution <: SniperDistribution 
    interps::Interpolants
end

function create_observation(pomdp::SniperPOMDP)
    interps = Interpolants(1)
    push!(interps, 1, 1.0)
    d = ObsDistribution(interps)
    return d
end

function observation!(d::ObsDistribution, pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    # if point is inside building can not see anything
    interps = d.interps
    invalids = pomdp.invalid_positions
    null_obs = pomdp.null_obs 
    if in(x, invalids) || in(y, invalids)
        interps.indices[1] = null_obs
    elseif x == y
        interps.indices[1] = y
    else
        # convert from idxs to (x,y) pos
        temp1 = pomdp.temp_position 
        temp2 = pomdp.temp_position2
        i2p!(temp1, pomdp, x)
        i2p!(temp2, pomdp, y)
        if isVisible(pomdp.map, temp1, temp2)
            # if the two points visible observation is position of y
            interps.indices[1] = y
        else
            # otherwise observation is null    
            interps.indices[1] = null_obs
        end
    end
    d
end


function reward(pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    agent = pomdp.agent
    if agent == :resource
        return resource_reward(pomdp, x, y, a)
    else
        return sniper_reward(pomdp, x, y, a)
    end
end
function resource_reward(pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    map = pomdp.map
    target = pomdp.target
    px = pomdp.temp_position 
    py = pomdp.temp_position2
    i2p!(px, pomdp, x)
    i2p!(py, pomdp, y)
    r = 0.0
    # movement penalty
    if a != 1
        r += pomdp.r_move
    end
    # shot reward
    if isVisible(map, px, py)    
        r += s_reward(pomdp, px, py) 
    end
    # obs reward 
    if isVisible(map, px, target)
        r += o_reward(pomdp, px, target)
    end
    return r
end
function sniper_reward(pomdp::SniperPOMDP, x::Int64, y::Int64, a::Int64)
    map = pomdp.map
    target = pomdp.target
    px = pomdp.temp_position 
    py = pomdp.temp_position2
    i2p!(px, pomdp, x)
    i2p!(py, pomdp, y)
    r = 0.0
    # movement penalty
    if a != 1
        r += pomdp.r_move
    end
    # shot reward
    if isVisible(map, px, py)    
        r += s_reward(pomdp, px, py) 
    end
    # obs reward 
    if isVisible(map, py, target)
        r += o_reward(pomdp, py, target)
    end
    return r
end
function s_reward(pomdp::SniperPOMDP, px::Vector{Int64}, py::Vector{Int64})
    return pomdp.r_shot 
end
function o_reward(pomdp::SniperPOMDP, px::Vector{Int64}, target::(Int64,Int64))
    return pomdp.r_obs
end


type VarSpace <: AbstractSpace
    var_iter::UnitRange{Int64}
end

fully_obs_space(pomdp::SniperPOMDP) = VarSpace(1:pomdp.x_size * pomdp.y_size)
part_obs_space(pomdp::SniperPOMDP) = VarSpace(1:pomdp.x_size * pomdp.y_size)


type ActionSpace <: AbstractSpace
    action_iter::UnitRange{Int64}
end

actions(pomdp::SniperPOMDP) = ActionSpace(1:n_actions(pomdp))
actions!(acts::ActionSpace, pomdp::SniperPOMDP, x::Int64, y::Int64) = acts

type ObservationSpace <: AbstractSpace
    obs_iter::UnitRange{Int64}
end

observations(pomdp::SniperPOMDP) = ObservationSpace(1:pomdp.null_obs)
observations!(obs::ObservationSpace, pomdp::SniperPOMDP, x::Int64, y::Int64) = obs


domain(space::VarSpace) = space.var_iter
domain(space::ActionSpace) = space.action_iter
domain(space::ObservationSpace) = space.obs_iter



function p2i(pomdp::SniperPOMDP, p::Vector{Int64})
    return sub2ind(pomdp.sizes, p)
end

function i2p!(p::Vector{Int64}, pomdp::SniperPOMDP, i::Int64)
    sizes = pomdp.sizes
    ind2sub!(p, sizes, i)
    p
end


function get_invalid_positions(map::Map, x_size::Int64, y_size::Int64)
    invalid = Set{Int64}() 
    sub_size = (x_size, y_size)
    for x = 1:x_size
        for y = 1:y_size
            p = (float(x),float(y))
            if inBuilding(map, p)
                pIdx = sub2ind(sub_size, x, y) 
                push!(invalid, pIdx)
            end
        end
    end
    return invalid
end



end # module
