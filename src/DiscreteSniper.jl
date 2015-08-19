module DiscreteSniper

export 
    SniperPOMDP,
    create_fully_obs_transition,
    create_partially_obs_transition,
    create_transition_distribution,
    create_observation_distribution,
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
    discount,
    move,
    move!,
    inbounds,
    i2xy!,
    xy2i,
    i2s!,
    s2i,
    i2p!,
    p2i

using MOMDPs
using POMDPToolbox

using Maps
using UrbanMaps
using Helpers # ind2sub!
using RayCasters # inBuilding


import MOMDPs: create_fully_obs_transition, create_partially_obs_transition, create_observation_distribution,
create_transition_distribution
import MOMDPs: weight, index, fully_obs_space, part_obs_space
import MOMDPs: n_states, n_actions, n_observations 
import MOMDPs: states, actions, actions!, observations, observations!, domain
import MOMDPs: reward, transition!, observation!
import MOMDPs: discount



type SniperPOMDP <: MOMDP

    x_size::Int64
    y_size::Int64
    grid_size::Int64
    point_sizes::Vector{Int64}
    state_sizes::Vector{Int64}
    sub_sizes::(Int64,Int64)
    aggrogate_sizes::Vector{Int64}

    n_actions::Int64

    r_obs::Float64
    r_shot::Float64
    r_move::Float64

    adversary_policy::Vector{Int64} # policy for adversary
    adversary_prob::Float64 # prob the adversary follows model
    invalid_positions::Set{Int64}

    temp_position::Vector{Int64}
    temp_position2::Vector{Int64}
    temp_state::Vector{Int64}
    
    target::(Int64,Int64)

    action_map::Matrix{Float64}

    null_obs::Int64

    agent::Symbol
    lvlk::Bool

    discount_factor::Float64

    map::Map

    # pre-allocated for memory efficiency

    function SniperPOMDP(map::Map; nSnipers::Int64 = 1, nMonitors::Int64 = 1,
                       target = (2,8),
                       target_reward = 0.1, sniper_reward = -1.0, move_reward = -0.01,
                       agent::Symbol = :resource, lvlk::Bool=false, 
                       adversary_policy::Vector{Int64}=zeros(Int64,0),
                       adversary_prob::Float64=0.5,
                       discount_factor::Float64=0.95)
        self = new()

        x_size = map.xSize
        y_size = map.ySize

        self.x_size = x_size
        self.y_size = y_size
        self.grid_size = x_size*y_size
        self.point_sizes = [x_size, y_size]
        self.state_sizes = [x_size, y_size, x_size, y_size]
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
        self.temp_state = [1,1,1,1]
        self.invalid_positions = get_invalid_positions(map, x_size, y_size)

        action_map = [0 1 -1 0 0 1 1 -1 -1; 0 0 0 1 -1 1 -1 1 -1]
        self.action_map = action_map 
        self.n_actions = size(action_map, 2)

        self.agent = agent # :resource or :sniper
        self.lvlk = lvlk

        self.adversary_policy = adversary_policy
        self.adversary_prob = adversary_prob


        self.discount_factor = discount_factor

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

# Aggrogate Distribution
type TransitionDistribution <: SniperDistribution
    fod::FODistribution
    pod::PODistribution
    interps::Interpolants
end

# currently interps are zero length
function create_fully_obs_transition(pomdp::SniperPOMDP)
    # fully observable variable moves deterministically
    interps = Interpolants(1)
    push!(interps, 1, 1.0)
    FODistribution(interps)
end

function create_partially_obs_transition(pomdp::SniperPOMDP)
    na = n_actions(pomdp)
    interps = Interpolants(na)
    p = 1.0 / na
    for i = 1:na; push!(interps, i, p); end;
    PODistribution(interps)
end

function create_transition_distribution(pomdp::SniperPOMDP)
    fod = create_fully_obs_transition(pomdp)
    pod = create_partially_obs_transition(pomdp)
    interps = deepcopy(pod.interps)
    TransitionDistribution(fod, pod, interps)
end


weight(d::SniperDistribution, i::Int64) = d.interps.weights[i]
index(d::SniperDistribution, i::Int64) = d.interps.indices[i]
Base.length(d::SniperDistribution) = d.interps.length

n_states(pomdp::SniperPOMDP) = (pomdp.x_size * pomdp.y_size)^2
n_actions(pomdp::SniperPOMDP) = pomdp.n_actions
n_observations(pomdp::SniperPOMDP) = pomdp.x_size * pomdp.y_size + 1 


# transition function for aggregate state index
# state = (xm, ym, xt, yt)
function transition!(d::TransitionDistribution, pomdp::SniperPOMDP, s::Int64, a::Int64)
    xy = pomdp.temp_position
    i2xy!(xy, pomdp, s)
    x = xy[1]; y = xy[2]
    transition!(d.fod, pomdp, x, y, a)
    transition!(d.pod, pomdp, x, y, a)
    # weights are the same as partially observable dist
    copy!(d.interps.weights, d.pod.interps.weights)
    xy[1] = d.fod.interps.indices[1]
    for i = 1:d.interps.length
        xy[2] = d.pod.interps.indices[i] 
        d.interps.indices[i] = xy2i(pomdp, xy)
    end
    d
end

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
    # transition stochastic
    stochastic_transition!(d, pomdp, x, y, a)
    # get the level-k policy
    interps = d.interps

    mu = pomdp.adversary_prob

    # if mu is less than stochastic transition, transition stochastic by default
    if mu <= interps.weights[1]
        return d
    end

    ag = aggrogate(pomdp, x, y)
    lvlk_action = pomdp.adversary_policy[ag]

    # the number of valid neighbors in uniformaly stochastic transition
    nvalids = 1.0/interps.weights[1]

    new_prob = (1.0 - mu) / (nvalids - 1.0)

    # fill the valid transitions with re-nromalized values
    for i = 1:length(d)
        w = weight(d, i)
        if w > 0.0
            interps.weights[i] = new_prob 
        end
    end

    # assign mu to state that lvl-k policy leads to
    ytemp = pomdp.temp_position
    move!(ytemp, pomdp, y, lvlk_action)
    yp = p2i(pomdp, ytemp)
    for i = 1:length(d)
        yidx = index(d, i)
        if yp == yidx
            interps.weights[i] = mu
            break
        end
    end
    d
end

function move!(p::Vector{Int64}, pomdp::SniperPOMDP, x::Int64, a::Int64)
    sizes = pomdp.point_sizes
    am = pomdp.action_map 
    ind2sub!(p, sizes, x) 
    p[1] += am[1,a] 
    p[2] += am[2,a] 
    p
end



function aggrogate(pomdp::SniperPOMDP, x::Int64, y::Int64)
    temp = pomdp.temp_position
    temp[1] = x
    temp[2] = y
    return xy2i(pomdp, temp) 
end


# Fully Observable Distribution
type ObsDistribution <: SniperDistribution 
    interps::Interpolants
end

function create_observation_distribution(pomdp::SniperPOMDP)
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


function reward(pomdp::SniperPOMDP, s::Int64, a::Int64)
    xy = pomdp.temp_position
    i2xy!(xy, pomdp, s)
    x = xy[1]; y = xy[2]
    reward(pomdp, x, y, a)
end

# TODO: add seperate reward quantities for resource and threat objectives
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
        r -= s_reward(pomdp, px, py) # rewards inversed
    end
    # obs reward 
    if isVisible(map, py, target)
        r -= o_reward(pomdp, py, target) # rewards inversed
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
states(pomdp::SniperPOMDP) = VarSpace(1:pomdp.x_size^2 * pomdp.y_size^2)

type ActionSpace <: AbstractSpace
    action_iter::UnitRange{Int64}
end

actions(pomdp::SniperPOMDP) = ActionSpace(1:n_actions(pomdp))
actions!(acts::ActionSpace, pomdp::SniperPOMDP, x::Int64, y::Int64) = acts
actions!(acts::ActionSpace, pomdp::SniperPOMDP, s::Int64) = acts

type ObservationSpace <: AbstractSpace
    obs_iter::UnitRange{Int64}
end

observations(pomdp::SniperPOMDP) = ObservationSpace(1:pomdp.null_obs)
observations!(obs::ObservationSpace, pomdp::SniperPOMDP, x::Int64, y::Int64) = obs


domain(space::VarSpace) = space.var_iter
domain(space::ActionSpace) = space.action_iter
domain(space::ObservationSpace) = space.obs_iter


discount(pomdp::SniperPOMDP) = pomdp.discount_factor


function p2i(pomdp::SniperPOMDP, p::Vector{Int64})
    return sub2ind(pomdp.point_sizes, p)
end

function i2p!(p::Vector{Int64}, pomdp::SniperPOMDP, i::Int64)
    sizes = pomdp.point_sizes
    ind2sub!(p, sizes, i)
    p
end

function s2i(pomdp::SniperPOMDP, s::Vector{Int64})
    return sub2ind(pomdp.state_sizes, s)
end

function i2s!(s::Vector{Int64}, pomdp::SniperPOMDP, i::Int64)
    ind2sub!(s, pomdp.state_sizes, i)
    s
end

function xy2i(pomdp::SniperPOMDP, xy::Vector{Int64})
    return sub2ind(pomdp.aggrogate_sizes, xy)
end

function i2xy!(xy::Vector{Int64}, pomdp::SniperPOMDP, i::Int64)
    ind2sub!(xy, pomdp.aggrogate_sizes, i)
    xy
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
