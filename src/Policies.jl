module Policies

export 
    Policy,
    AlphaPolicy,
    MCMCPolicy,
    RandomPolicy,
    SarsopPolicy,
    getOfflineAction,
    getBestState


using ReadPolicy
using MOMDPs

# this is taken from Louis Dressel's POMDPs.jl. Thanks Louis!

###########################################################################
# policy.jl
# Introduces the policy abstract type and various subclasses
###########################################################################
#import Distributions: entropy

abstract Policy

typealias Belief Vector{Float64}

###########################################################################
# ALPHA POLICY
# Each row of alpha_vectors is an alpha_vector
###########################################################################
# TODO: Give alpha_actions a type
#  I think so far, it will most likely be a Vector{Int}
#  QMDP, FIB, SARSOP all produce something amenable to this
#  I just have to make sure I don't assume it is something else later
type AlphaPolicy <: Policy

    # Each row of alpha_vectors is an alpha_vector
    model::POMDP
    alpha_vectors::Matrix{Float64}
    alpha_actions::Vector{Int64}
    observable_states::Vector{Int64}

    # I think you need this if you provide an additional constructor
    # If you add another constructor, it overwrites the default
    #  even if it has a different signature
    # So you gotta add it back in there
    AlphaPolicy(m::POMDP, av::Matrix{Float64}, aa::Vector{Int64}, os::Vector{Int64}) = new(m, av, aa, os)

    # Constructor if no action list is given
    # Here, we 0-index actions, to match sarsop output
    function AlphaPolicy(m::POMDP, av::Matrix{Float64})
        numActions = size(av, 1)
        alist = [0:(numActions-1)]
        ostates = [0:(numActions-1)]
        return new(m, av, alist, ostates)
    end

    # Constructor reading policy from file
    function AlphaPolicy(m::POMDP, filename::String)
        alpha_vectors, alpha_actions, observable_states = readpolicy(filename)
        return new(m, alpha_vectors, alpha_actions, observable_states)
    end
end



# Returns the the action to take
# b is a belief state
function getOfflineAction(policy::AlphaPolicy, b::Belief)

    bSize = size(policy.alpha_vectors, 2)
    if length(b) == bSize
        utilities = policy.alpha_vectors * b
    elseif length(b) > bSize
        bp = trimBelief(b, bSize)
        utilities = policy.alpha_vectors * bp
    else 
        println("Alpha and belief size mismatch")
    end

    #utilities = policy.alpha_vectors * s.beliefs[t-1]

    # DEBUG
    #println("utilities = $(round(utilities,3))")

    # following is done if alpha_actions is array of ints
    actionIndex = policy.alpha_actions[indmax(utilities)] + 1
    #return policy.model.actions[actionIndex]
    return actionIndex
end

function getOfflineAction(policy::AlphaPolicy, b::Belief, obsState::Int)

    vectors = policy.alpha_vectors
    actions = policy.alpha_actions
    states = policy.observable_states
    o = obsState - 1 # julia obs: 1-100, sarsop obs: 0-99

    bSize = size(policy.alpha_vectors)[2]
    if length(b) == bSize
        utilities = vectors * b
    elseif length(b) > bSize
        bp = trimBelief(b, bSize)
        utilities = vectors * bp
    else 
        println("Alpha and belief size mismatch")
    end

    aChunk = actions[find(s -> s == o, states)]
    actionIndex = aChunk[indmax(utilities[find(s -> s == o, states)])] + 1
    return actionIndex
end

function trimBelief(b::Belief, bSize::Int)
    idxs = find(b)
    sIdx = minimum(idxs) - minimum(idxs) % bSize + 1
    eIdx = sIdx + bSize - 1
    return b[sIdx:eIdx]
end




###########################################################################
# RANDOM POLICY
###########################################################################

type RandomPolicy <: Policy

    pomdp::POMDP

    function RandomPolicy(pm::POMDP)
        self = new()
        self.pomdp = pm
        return self
    end
end


function getOfflineAction(policy::RandomPolicy, b::Belief, obsState::Int)

    pomdp    = policy.pomdp
    nActions = length(pomdp.actions)

    mPos = pomdp.positions[obsState]

    s = State([1., 1., mPos.x, mPos.y, 1])
    si = pomdp.x2i(pomdp, s)

    feasibleActions = Array(Int64, 0)

    for a = 1:nActions
        nextStates = pomdp.getNextStates(pomdp, si, a)[1]
        length(nextStates) > 0 ? push!(feasibleActions, a) : nothing
    end

    ap = feasibleActions[rand(1:end)]

    return ap
end
        





###########################################################################
# SARSOP POLICY
###########################################################################
# This version exists separately from alphapolicy because
#  in a momdp, alphavectors can be associated with observed states
# TODO: Consolidate this with alpha policy. Create alpha vector object?
#  Maybe you can override multiplication function for these vectors?
type SarsopPolicy <: Policy

    # Each row of alpha_vectors is an alpha_vector
    model::POMDP
    alpha_vectors::Matrix{Float64}
    alpha_actions::Vector{Int}
    alpha_obs::Vector{Int}

    # not sure if I need this line
    SarsopPolicy(model, alpha_vectors, alpha_actions, alpha_obs) = new(model, alpha_vectors, alpha_actions, alpha_obs)

    # Not sure I particularly care for this constructor
    # I don't think model.name is going to be accurate
    # Best name the file itself
    function SarsopPolicy(model::POMDP)
        alpha_vectors, alpha_actions, alpha_obs = readpolicy(model.name)
        return new(model, alpha_vectors, alpha_actions, alpha_obs)
    end

    function SarsopPolicy(model::POMDP, filename::ASCIIString)
        alpha_vectors, alpha_actions, alpha_obs = readpolicy(filename)
        return new(model, alpha_vectors, alpha_actions, alpha_obs)
    end
end

# The sarsop policies are generated assuming the state of the vehicle is observable
function getOfflineAction(policy::SarsopPolicy, b::Belief)

    gridSize = policy.model.gridSize
    numStates = policy.model.numStates

    # Determine xv and yv from the belief
    s = [0,0,0,0]
    for i = 1:numStates
        if b[i] > 0.
            s = ind2state(policy.model, i)
            break
        end
    end
    xv = s[1]
    yv = s[2]

    # Convert this xv and yv to an obsValue
    # TODO: This might be in reverse order
    # Note that c++ is row-major order
    # Julia is column-major order
    obsValue = sub2ind((gridSize, gridSize), yv+1, xv+1) - 1

    # Convert b to 4x4, and take the slice representing 
    bnew = deepcopy(reshape(b, gridSize, gridSize, gridSize, gridSize))
    bnew = bnew[xv+1, yv+1, :, :]
    bnew = reshape(bnew, gridSize, gridSize)'
    bnew = reshape(bnew, gridSize*gridSize, 1)

    # Determine the maximum utility that is of the same obsvalue
    utilities = policy.alpha_vectors * bnew
    numVectors = length(utilities)
    maxUtility = utilities[1]
    bestAction = policy.alpha_actions[1]
    for i = 1:numVectors
        if policy.alpha_obs[i] == obsValue
            if utilities[i] > maxUtility
                maxUtility = utilities[i]
                bestAction = policy.alpha_actions[i]
            end
        end
    end

    # following is done if alpha_actions is array of ints
    if typeof(policy.alpha_actions) == Array{Int,1}
        actionIndex = bestAction + 1
        return policy.model.actions[actionIndex]
    else
        #println("indmax(utilities) = $(indmax(utilities))")
        return bestAction
    end
end



end # module
