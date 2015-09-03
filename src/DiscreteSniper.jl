module DiscreteSniper

export 
    SniperPOMDP,
    create_state,
    create_fully_obs_transition,
    create_partially_obs_transition,
    create_transition_distribution,
    create_observation_distribution,
    weight, 
    index, 
    fully_obs_space,
    part_obs_space,
    states,
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
    transition,
    observation!,
    discount,
    rand,
    # misc
    move,
    move!,
    inbounds,
    i2xy!,
    xy2i,
    i2s!,
    s2i,
    i2p!,
    p2i,
    aggrogate,
    valid_action,
    isVisible,
    get_visibles,
    get_ballistics,
    # space
    VarSpace,
    ActionSpace,
    ObservationSpace


using MOMDPs
using POMDPToolbox

using Maps
using UrbanMaps
using Helpers # ind2sub!
using RayCasters # inBuilding
using BallisticModels


import MOMDPs: create_fully_obs_transition, create_partially_obs_transition, create_observation_distribution,
create_transition_distribution, create_state
import MOMDPs: weight, index, fully_obs_space, part_obs_space
import MOMDPs: n_states, n_actions, n_observations 
import MOMDPs: states, actions, actions!, observations, observations!, domain
import MOMDPs: reward, transition!, observation!
import MOMDPs: discount
import RayCasters: isVisible


const LINEAR_BALLISTICS = [1.0,-0.5]
const FOURTH_ORDER_BALLISTICS = [1.0, -0.886851, 0.15398, 0.0460204, -0.0131487]

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
    temp_position3::Vector{Int64}
    temp_state::Vector{Int64}
    
    target::(Float64,Float64)

    action_map::Matrix{Float64}

    null_obs::Int64

    agent::Symbol # :resource or :sniper
    lvlk::Bool

    discount_factor::Float64

    map::Map

    visibles::Matrix{Bool}

    ballistic_model::BallisticModel
    obs_model::BallisticModel
    ballistics::Matrix{Float64}
    obs::Vector{Float64}

    # pre-allocated for memory efficiency

    function SniperPOMDP(map::Map; nSnipers::Int64 = 1, nMonitors::Int64 = 1,
                       target = (0.5,0.5), # in normalized cooridnates
                       target_reward = 0.1, sniper_reward = -1.0, move_reward = -0.01,
                       agent::Symbol = :resource, lvlk::Bool=false, 
                       adversary_policy::Vector{Int64}=zeros(Int64,0),
                       adversary_prob::Float64=0.3,
                       discount_factor::Float64=0.95,
                       ballistic_model::BallisticModel=SimplePoly(LINEAR_BALLISTICS),
                       obs_model::BallisticModel=Constant())
        @assert agent == :resource || agent == :sniper "Invalid agent type"

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

        target_coords = (x_size*target[1], y_size*target[2])
        self.target = target_coords

        self.map = map

        self.temp_position = [1,1]
        self.temp_position2 = [1,1]
        self.temp_position3 = [1,1]
        self.temp_state = [1,1,1,1]
        self.invalid_positions = get_invalid_positions(map, x_size, y_size)

        action_map = [0 1 -1 0 0 1 1 -1 -1; 0 0 0 1 -1 1 -1 1 -1]
        self.action_map = action_map 
        self.n_actions = size(action_map, 2)

        self.agent = agent # :resource or :sniper
        self.lvlk = lvlk

        self.adversary_policy = adversary_policy
        self.adversary_prob = adversary_prob

        self.visibles = get_visibles(map, x_size, y_size)

        self.ballistic_model = ballistic_model
        self.obs_model = obs_model
        self.ballistics = get_ballistics(map, ballistic_model, x_size, y_size)
        #self.obs = get_ballistics(map, obs_model, x_size, y_size)

        self.discount_factor = discount_factor

        return self
    end
end

# returns a random initial state
function create_state(pomdp::SniperPOMDP)
    invalids = pomdp.invalid_positions
    p1 = pomdp.temp_position
    p2 = pomdp.temp_position2
    s1 = rand(1:pomdp.grid_size)
    s2 = rand(1:pomdp.grid_size)
    i2p!(p1, pomdp, s1); i2p!(p2, pomdp, s2)
    while isVisible(pomdp.map, p1, p2) || in(s1, invalids) || in(s2, invalids)
        s1 = rand(1:pomdp.grid_size)
        s2 = rand(1:pomdp.grid_size)
        i2p!(p1, pomdp, s1); i2p!(p2, pomdp, s2)
    end
    s = aggrogate(pomdp, s1, s2)
    return s
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


# transition function for aggrogate state index
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
#function transition(pomdp::SniperPOMDP, s::Int64, a::Int64; d=create_transition_distribution(pomdp))
#    transition!(d, pomdp, s, a)
#end
function transition(pomdp::SniperPOMDP, s::Int64, a::Int64, d=create_transition_distribution(pomdp))
    transition!(d, pomdp, s, a)
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
        if in(xp, invalids) || !inbounds(pomdp.map, temp) || !isVisible(pomdp, x, xp)
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
            if in(yp, pomdp.invalid_positions) || !inbounds(pomdp.map, temp) || !isVisible(pomdp, y, yp)
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

    # NOTE: aggrogate is over (x,y) not (y,x)
    # The policies are generated for the adversary (y var), so we want the action for their state config
    ag = aggrogate(pomdp, y, x)
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

#=
function move!(p::Vector{Int64}, pomdp::SniperPOMDP, x::Int64, a::Int64)
    sizes = pomdp.point_sizes
    am = pomdp.action_map 
    pp = pomdp.temp_position3
    # find the point of index x
    ind2sub!(p, sizes, x) 
    # copy to temp position
    copy!(pp, p)
    # move temp position
    pp[1] += am[1,a] 
    pp[2] += am[2,a] 
    if !inbounds(pomdp.map, pp)
        # dont move if out of bounds
        return p
    end
    xp = p2i(pomdp, pp)
    # only move p to temp position if that point is visible, otherwise stay 
    if isVisible(pomdp, x, xp)
        copy!(p, pp)
    end
    p
end
=#

function move!(p::Vector{Int64}, pomdp::SniperPOMDP, x::Int64, a::Int64)
    sizes = pomdp.point_sizes
    am = pomdp.action_map 
    # find the point of index x
    ind2sub!(p, sizes, x) 
    # move position
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
        if isVisible(pomdp, x, y)
            # if the two points visible observation is position of y
            interps.indices[1] = y
        else
            # otherwise observation is null    
            interps.indices[1] = null_obs
        end
    end
    d
end


function Base.rand(d::ObsDistribution)
    return d.interps.indices[1]
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
    #if isVisible(pomdp, x, y)    
    #    r += s_reward(pomdp, px, py) 
    #end
    r += s_reward(pomdp, x, y)
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
    #=
    if isVisible(pomdp, x, y)    
        r -= s_reward(pomdp, px, py) # rewards inversed
    end
    =#
    r -= s_reward(pomdp, x, y)
    # obs reward 
    if isVisible(map, py, target)
        r -= o_reward(pomdp, py, target) # rewards inversed
    end
    return r
end
function s_reward(pomdp::SniperPOMDP, px::Vector{Int64}, py::Vector{Int64})
    return pomdp.r_shot 
end
function s_reward(pomdp::SniperPOMDP, x::Int64, y::Int64)
    return pomdp.r_shot * pomdp.ballistics[x,y]
end
function o_reward(pomdp::SniperPOMDP, px::Vector{Int64}, target::(Int64,Int64))
    return pomdp.r_obs
end
function o_reward(pomdp::SniperPOMDP, px::Vector{Int64}, target::(Float64,Float64))
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

function isVisible(pomdp::SniperPOMDP, x::Int64, y::Int64)
    return pomdp.visibles[x,y]
end

# check if action a is valid from point index p
function valid_action(pomdp::SniperPOMDP, p::Int64, a::Int64)
    invalids = pomdp.invalid_positions 
    pos = pomdp.temp_position
    move!(pos, pomdp, p, a)
    pp = p2i(pomdp, pos)
    # if an invalid state then action is invalid
    if in(pp, invalids) || !inbounds(pomdp.map, pos)
        return false
    end
    return true
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

function get_visibles(map::Map, xs::Int64, ys::Int64)
    npts = xs*ys
    visibles = zeros(Bool, npts, npts)
    p1 = [1,1]
    p2 = [1,1]
    sizes = [xs, ys]
    for i = 1:npts, j = 1:npts
        ind2sub!(p1, sizes, i)
        ind2sub!(p2, sizes, j)
        isVisible(map, p1, p2) ? (visibles[i,j] = true) : (visibles[i,j] = false)
    end
    return visibles
end

function get_ballistics(map::Map, m::BallisticModel, xs::Int64, ys::Int64)
    npts = xs*ys
    ballistics = zeros(npts, npts)
    p1 = [1,1]
    p2 = [1,1]
    sizes = [xs, ys]
    for i = 1:npts, j = 1:npts
        ind2sub!(p1, sizes, i)
        ind2sub!(p2, sizes, j)
        if isVisible(map, p1, p2)
            ballistics[i,j] = prob(m, p1, p2, xs, ys)  
        end
    end
    return ballistics
end

end # module
