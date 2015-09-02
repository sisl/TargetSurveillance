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
    map_shift::Float64
end
function SniperServer(socket::Int64; delay::Int64=0, protocol=Dict(), shift::Float64=0.0)
    if isempty(protocol)
        protocol[:next] = "next"
        protocol[:null] = "NULL"
        protocol[:start_game] = "start"
        protocol[:end_game] = "end"
        protocol[:kill] = "kill"
        protocol[:initial] = "initial"
    end

    return SniperServer(socket, delay, protocol, shift)
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
    belief_values::Vector{Float64}
    shift::Float64
    a::Int64 # for belief updating
end
function ClientResults(;flag::String="split", shift::Float64=0.0)
    rounders = [[0.0, 0.0] [-0.5, -0.5] [0.5, -0.5] [-0.5, 0.5] [0.5, 0.5]]
    np = [-1,-1]
    n1 = zeros(Int64,5)
    n2 = zeros(Float64,5)
    return ClientResults(flag,rounders,"","","",[1,1],[1,1],[1.,1.],[1.,1.],np,n1,n2,shift,1)
end
raw_string(r::ClientResults) = r.s
pos(r::ClientResults) = r.posf
obs(r::ClientResults) = r.of
position_neighbor(r::ClientResults) = r.posi
observation_neighbor(r::ClientResults) = r.oi
null_position(r::ClientResults, p::Vector{Int64}) = r.null_position == p
threat_neighbor_indexes(r::ClientResults) = r.threat_neighbor_idxs
threat_belief_values(r::ClientResults) = r.belief_values

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
    threati = r.threat_neighbor_idxs
    bvals = r.belief_values
    # assign values to these
    oi = r.oi # observation neighbors
    posi = r.posi # position neighbors
    of = r.of # observation neighbors
    posf = r.posf # position neighbors
    #################################################################
    # fill the position arrays
    posf[1:end] = float(split(ps))[1:end] + r.shift
    nearest_neighbor!(posi, pomdp, posf, rounders, threati, bvals, false)
    # fill the observaion arrays
    if os == server.protocol[:null]
        fill!(oi,-1)
        fill!(of,-1.0)
    else
        of[1:end] = float(split(os))[1:end] + r.shift
   #     nearest_neighbor!(oi, pomdp, of, posi, rounders, threati, bvals)
        nearest_neighbor!(oi, pomdp, of, rounders, threati, bvals, true)
    end
    r
end

function nearest_neighbor!(pp::Vector{Int64}, pomdp::POMDP, p::Vector{Float64}, rounders::Matrix{Float64}, threat_neighbor_idxs::Vector{Int64}, belief_values::Vector{Float64}, checkingThreat::Bool)
    # find the closest valid neighbor and if looking neighbors for threat fill in the threat neighbor matrix
    # re-scale to map size
    p[1] = pomdp.x_size*p[1]; p[2] = pomdp.y_size*p[2]
    pts = zeros(Int64,2)
    closest = Inf
    numOfValidPoints = -1
    for i = 1:5
        pts[1:end] = int(p+rounders[:,i])[1:end]
        d = sqrt((pts[1]-p[1])^2 + (pts[2]-p[2])^2)
        # find the distance to the each corner
        idx = p2i(pomdp, pts)
        if !in(idx, pomdp.invalid_positions) && inbounds(pomdp.map, pts)
            if d < closest
                pp[1:end] = pts[1:end]
                closest = d
            end
            if checkingThreat
                threat_neighbor_idxs[i] = idx
                numOfValidPoints += 1
            end
        else
            threat_neighbor_idxs[i] = -1
        end
    end
    fill!(belief_values, 1/numOfValidPoints)
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
    res = ClientResults(shift=sserver.map_shift)
    while true
        conn = accept(server)
        @async begin
        try
            println("Connected")
            while true
            #    println("Waiting to read")
                line = readline(conn)
                line = strip(line)
           #     println("From Client: ", line)

                #if line == protocol[:initiall

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
                    # fill initials
                    res.a = 1
                    println("Initial Positions: \nResource: $mp \nThreat: $sp")
                    println("Initial Belief: $(b.b)")

                # update belief and send way point info
                elseif line == protocol[:next]
                    tic()
                  #  println("Sending waypoint")
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
                    update_belief!(b, pomdp, mi, res.a, si)
                    a = action(policy, b, mi)
                    res.a = deepcopy(a)
                    #update_belief!(b, pomdp, mi, a, si)
                    threatIndexes = threat_neighbor_indexes(res)
                    beliefValues = threat_belief_values(res)
                    valid(b) ? nothing : fill!(b, threatIndexes, beliefValues)
                    # find the new waypoint
                    move!(p, pomdp, mi, a)
                    tp = p - res.shift # for maps that are shifted
                    waypoint = "$(tp[1]/pomdp.x_size) $(tp[2]/pomdp.y_size)\n"
                    write(conn, waypoint)
                    println("Recieved: $line")
                    println("Positions: \nResource: $mp \nThreat: $sp")
                    println("Waypoint: $waypoint")
                    println("Belief: $(b.b)")
                    toc()

                # end game
                elseif line == protocol[:end_game]
                    println("Ending the Game")
                    b = DiscreteBelief(ns)
                    break
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
