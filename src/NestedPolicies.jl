module NestedPolicies

export
    NestedPolicy,
    NestedSolver,
    solve!,
    get_policy,
    file_name,
    write

using DiscreteSniper
using DiscreteValueIteration
using HDF5, JLD
using MOMDPs

import MOMDPs: Policy, Solver, solve!

type NestedPolicy <: Policy
    file::ASCIIString
    own_policies::Vector{ValueIterationPolicy}
    adv_policies::Vector{ValueIterationPolicy}
    probs::Vector{Float64} # here in case needed later
    # default constructor - first part of file until -level.nested
    function NestedPolicy(file::ASCIIString)
        self = new()
        if isfile(file) 
            # load policies
            d = load(file)["policy"]
            od = d["own"]
            ad = d["adv"]
            ol = length(od); al = length(ad)
            op = Array(ValueIterationPolicy, ol)
            ap = Array(ValueIterationPolicy, al)
            for i = 1:ol
                (q, u, p, am) = locals(od[i])
                op[i] = ValueIterationPolicy(q, u, p, am)
            end
            for i = 1:al
                (q, u, p, am) = locals(ad[i])
                ap[i] = ValueIterationPolicy(q, u, p, am)
            end
            probs = Float64[]
            self.file = file
            self.own_policies = op
            self.adv_policies = ap
            self.probs = probs
            return self
        else
            op = ValueIterationPolicy[]
            ap = ValueIterationPolicy[]
            probs = Float64[]
            self.file = file
            self.own_policies = op
            self.adv_policies = ap
            self.probs = probs
            return self
        end
    end
end

type NestedSolver <: Solver
    k::Int64
    n_iter::Int64
    eps::Float64
    verbose::Bool
end
function NestedSolver(;k::Int64=2, n_iter::Int64=30, eps::Float64=1e-3, verbose::Bool=true)
    return NestedSolver(k, n_iter, eps, verbose)
end

function locals(d::Dict)
    return (d["Q"], d["utility"], d["policy"], d["action map"])
end

# returns the policy for the given reasoning level (k)
# given the agent (1 = own), (2 = adversary)
function get_policy(policy::NestedPolicy, k::Int64, agent::Int64)
    k += 1
    n = length(policy.own_policies)
    @assert k <= n "Can not extract level-$k policy, only level-$(n-1) available"
    agent == 2 ? (return policy.adv_policies[k]) : nothing
    return policy.own_policies[k]
end

# generates a policy file name for the sniper pomdp problem
function file_name(path::ASCIIString, k::Int64, policy::NestedPolicy, pomdp::SniperPOMDP)
    xs = pomdp.x_size
    ys = pomdp.y_size
    mu = pomdp.adversary_prob
    name = "$(path)nested-$(xs)x$(ys)-mu-$(mu)-level-$(k).nested"
    return name
end

# computes the nested policies for the sniper pomdp using value iteration
function solve!(policy::NestedPolicy, solver::NestedSolver, model::SniperPOMDP; dump::Bool=true)
    n_iter = solver.n_iter
    eps = solver.eps
    verbose = solver.verbose
    k = solver.k
    mu = model.adversary_prob
    map = deepcopy(model.map)

    own = policy.own_policies
    adv = policy.adv_policies
    empty!(own); empty!(adv)

    # level-0 own
    pomdp = SniperPOMDP(map, agent=:resource)
    po = ValueIterationPolicy(pomdp)
    solver = ValueIterationSolver(n_iter, eps)
    solve!(po, solver, pomdp, verbose=true)
    push!(own, deepcopy(po))
    # level-0 adversary
    pomdp = SniperPOMDP(map, agent=:sniper)
    pa = ValueIterationPolicy(pomdp)
    solver = ValueIterationSolver(n_iter, eps)
    solve!(pa, solver, pomdp, verbose=true)
    push!(adv, deepcopy(pa))

    for i = 1:k
        verbose ? println("\nStarting: Level-$i\n") : nothing
        # find level-i resource policy
        p = adv[i].policy # the level k-1 policy
        pomdp = SniperPOMDP(map, adversary_policy=p, adversary_prob=mu, lvlk=true, agent=:resource)
        po = ValueIterationPolicy(pomdp)
        solve!(po, solver, pomdp, verbose=true)
        push!(own, deepcopy(po))
        # find level-i sniper policy 
        p = own[i].policy # level k-1 policy
        pomdp = SniperPOMDP(map, adversary_policy=p, adversary_prob=mu, lvlk=true, agent=:sniper)
        pa = ValueIterationPolicy(pomdp)
        solve!(pa, solver, pomdp, verbose=true)
        push!(adv, deepcopy(pa))
        # write to file
        dump ? (write(policy, overwrite=true)) : (nothing)
    end
    policy
end

# returns the "own" and the "adv" policy sizes
function Base.size(d::Dict)
    os = -Inf
    as = -Inf
    for k in keys(d)
        s = split(k)
        t = s[1] # "own" or "adv"
        i = int(s[2]) # number in array
        if t == "own" && i > os
            os = i
        elseif t == "adv" && i > as
            as = i
        else
            continue
        end
    end
    return (os, as)
end

# writes policy results to files
function Base.write(policy::NestedPolicy; overwrite::Bool=false)
    file = policy.file

    # if file exists, overwrite flag must be set to true to write new policy
    isfile(file) && !overwrite ? (return) : nothing

    own = policy.own_policies
    adv = policy.adv_policies

    d = Dict()
    d["own"] = Dict[]
    d["adv"] = Dict[]
    # add own policies
    for i = 1:length(own)
        nd = Dict()
        fill!(nd, own[i])
        push!(d["own"], deepcopy(nd))
    end
    # add adv policies 
    for i = 1:length(adv)
        nd = Dict()
        fill!(nd, adv[i])
        push!(d["adv"], deepcopy(nd))
    end
    save(file, "policy", d)
end

function Base.fill!(d::Dict, p::ValueIterationPolicy)
    d["Q"] = p.qmat
    d["utility"] = p.util
    d["action map"] = p.action_map
    d["policy"] = p.policy
    d
end

end # module
