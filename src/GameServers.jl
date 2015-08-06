# interface for client-server communication 
# used for sending commands to a game
module GameServers

# interface with the simulator
using MOMDPs
using POMDPToolbox
using SARSOP
using DiscreteSniper
using RayCasters

export 
    # Types
    GameServer,
    SniperServer,
    ClientResults,
    # Sniper Server methods
    start,
    server_socket,
    # ClientResults methods
    pos,
    obs,
    position_neighbor,
    observation_neighbor,
    null_position,
    fill!,
    parse!,
    nearest_neighbor


abstract GameServer


type SniperServer <: GameServer
    socketNumber::Int64 
    sendDelay::Int64 # in ms
    protocol::Dict{Symbol, String} # desired operation to protocol
end
function SniperServer(socket::Int64; delay::Int64=0, protocol=Dict())
    if isempty(protocol)
        protocol[:next] = "next"
        protocol[:null] = "NULL"
        protocol[:start_game] = "start"
        protocol[:end_game] = "end"
        protocol[:kill] = "kill"
    end

    return SniperServer(socket, delay, protocol)
end
server_socket(s::SniperServer) = s.socketNumber


type ClientResults
    flag::String
    rounders::Matrix{Float64} # for finding nearest neighbor
    s::String
    p1::String
    p2::String
    posi::Vector{Int64}
    oi::Vector{Int64}
    posf::Vector{Float64}
    of::Vector{Float64}
    null_position::Vector{Int64}
    threat_neighbor_idxs::Vector{Int64}
    resource_neighbor_idxs::Vector{Int64}
end
function ClientResults(;flag::String="split")
    rounders = [[0.0, 0.0] [-0.5, -0.5] [0.5, -0.5] [-0.5, 0.5] [0.5, 0.5]]
    np = [-1,-1]
    n1 = zeros(Int64,4)
    n2 = zeros(Int64,4)
    return ClientResults(flag,rounders,"","","",[1,1],[1,1],[1.,1.],[1.,1.],np,n1,n2)
end
raw_string(r::ClientResults) = r.s
pos(r::ClientResults) = r.posf
obs(r::ClientResults) = r.of
position_neighbor(r::ClientResults) = r.posi
observation_neighbor(r::ClientResults) = r.oi
null_position(r::ClientResults, p::Vector{Int64}) = r.null_position == p
function Base.fill!(r::ClientResults, s::String)
    r.s = s
    t = split(s, r.flag)
    r.p1 = strip(t[1])
    r.p2 = strip(t[2])
    r
end
function parse!(r::ClientResults, pomdp::POMDP, server::SniperServer)
    rounders = r.rounders
    # position = (x,y) of resource
    # observation = (x,y) of threat (or NULL)
    ps = r.p1 # position string
    os = r.p2 # observation string
    # assign values to these
    oi = r.oi # observation neighbors
    posi = r.posi # position neighbors
    of = r.of # observation neighbors
    posf = r.posf # position neighbors
    ################################################################# 
    # fill the position arrays
    posf[1:end] = float(split(ps))[1:end]
    nearest_neighbor!(posi, pomdp, posf, rounders)
    # fill the observaion arrays
    if os == server.protocol[:null]
        fill!(oi,-1)
        fill!(of,-1.0)
    else
        of[1:end] = float(split(os))[1:end]
        nearest_neighbor!(oi, pomdp, of, posi, rounders)
    end
    r
end

function nearest_neighbor!(r::ClientResults, pomdp::POMDP)
    # finds the nearest neighbor and fills neighbor matrices
end

function nearest_neighbor!(pp::Vector{Int64}, pomdp::POMDP, p::Vector{Float64}, rounders::Matrix{Float64})
    # find the closest valid neighbor 
    # re-scale to map size
    p[1] = pomdp.x_size*p[1]; p[2] = pomdp.y_size*p[2]
    pts = zeros(Int64,2)
    closest = Inf
    for i = 1:5
        pts[1:end] = int(p+rounders[:,i])[1:end]
        d = sqrt((pts[1]-p[1])^2 + (pts[2]-p[2])^2)
        # find the distance to the each corner
        idx = p2i(pomdp, pts)
        if !in(idx, pomdp.invalid_positions) && d < closest && inbounds(pomdp.map, pts)
            pp[1:end] = pts[1:end]
            closest = d
        end
    end
    pp
end
function nearest_neighbor!(pp::Vector{Int64}, pomdp::POMDP, op::Vector{Float64}, sp::Vector{Int64}, rounders::Matrix{Float64})
    # check if the observation is possible from current position
    println("TEST")
    nearest_neighbor!(pp, pomdp, op, rounders)
    if isVisible(pomdp.map, pp, sp)
        return pp
    end
    # find the closest observation position that is visible
    fill!(pp,-1)
    pp
end




function start(sserver::SniperServer, pomdp::POMDP, policy::Policy)
    protocol = sserver.protocol
    map      = pomdp.map
    sn     = server_socket(sserver)
    server = listen(sn)
    println("Server Ready")
    p = [1,1]
    ns = length(collect(domain(part_obs_space(pomdp))))
    b = DiscreteBelief(ns)
    res = ClientResults()
    while true
        conn = accept(server)
        @async begin
        try
            println("Connected")
            while true
                println("Waiting to read")
                line = strip(readline(conn))
                println("From Client: ", line)
    
                # begin the game
                if line == protocol[:start_game]
                    println("Starting the game")
                    line = readline(conn) # reads tcp socket
                    fill!(res, line) # fills and splits the results
                    parse!(res, pomdp, sserver) # parses the results 
                    # get closest state positions on the pomdp grid
                    mp = position_neighbor(res) # returns the closest monitor position
                    sp = observation_neighbor(res) # returns the closest threat position
                    # check if sniper is initially observed and fill belief accordingly 
                    if null_position(res, sp)
                        # uniform belief
                        fill!(b, 1.0/ns)
                    else
                        # localized belief on a single state
                        si = p2i(pomdp, sp)
                        fill!(b, 0.0)
                        b[si] = 1.0
                    end
                    waypoint = "$(mp[1]/pomdp.x_size) $(mp[2]/pomdp.y_size)\n"
                    write(conn, waypoint) 
                    println("Initial Positions: \nResource: $mp \nThreat: $sp")
                    println("Initial Belief: $(b.b)")

                # update belief and send way point info
                elseif line == protocol[:next]
                    println("Sending waypoint")
                    line = readline(conn)
                    fill!(res, line)
                    parse!(res, pomdp, sserver) # parses the results 
                    # get closest state positions on the pomdp grid
                    mp = position_neighbor(res) # returns the closest monitor position
                    sp = observation_neighbor(res) # returns the closest threat position
                    si = 0
                    if null_position(res, sp)
                        si = pomdp.null_obs 
                    else
                        si = p2i(pomdp, sp)
                    end
                    # find optimal action and updated belief
                    mi = p2i(pomdp, mp)
                    a = action(policy, b, mi)
                    update_belief!(b, pomdp, mi, a, si)
                    #valid(b) ? nothing : fill!(b, idxs, vals)
                    # find the new waypoint
                    move!(p, pomdp, mi, a)
                    waypoint = "$(p[1]/pomdp.x_size) $(p[2]/pomdp.y_size)\n"
                    write(conn, waypoint) 
                    println("Positions: \nResource: $mp \nThreat: $sp")
                    println("Waypoint: $waypoint")
                    println("Belief: $(b.b)")

                # end game
                elseif line == protocol[:end_game]
                    println("Ending the Game")
                    b = DiscreteBelief(ns)
                elseif line == protocol[:kill]
                    println("Killing the server")
                    close(conn)
                    return nothing
                else
                    println("Invalid Protocol String")
                end
            end
        catch err
            print("Connection ended with error $err")
            close(conn)
        end
        end #async
    end
end


end # module
