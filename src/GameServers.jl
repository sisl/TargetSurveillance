# interface for client-server communication 
# used for sending commands to a game
module GameServers

# interface with the simulator
using MOMDPs
using POMDPToolbox
using SARSOP
using DiscreteSniper

export GameServer, SniperServer
export start, send, parse_results, nearest_neighbor!


abstract GameServer


type SniperServer <: GameServer
    socketNumber::Int64 
    sendDelay::Int64 # in ms
    protocol::Dict{Symbol, String} # desired operation to protocol
end
function SniperServer(socket::Int64; delay::Int64=0, protocol=Dict())
    ending = "\n"
    @windows_only ending = "\r"*ending
    if isempty(protocol)
        protocol[:next] = "next"*ending
        protocol[:null] = "NULL"*ending
        protocol[:start_game] = "start"*ending 
        protocol[:end_game] = "end"*ending
        protocol[:kill] = "kill"*ending
    end

    return SniperServer(socket, delay, protocol)
end
function serverSocket(server::SniperServer)
    return server.socketNumber
end

type GameParams
    rounders::Matrix{Float64}
end

type ClientResults
    flag::String
    s::String
    p1::String
    p2::String
    pos::Vector{Int64}
    o::Vector{Int64}
end
function ClientResults(;flag::String="split")
    return ClientResults(flag,"","","",[1,1],[1,1])
end
function get(r::ClientResults)
    return r.s
end
function position(r::ClientResults)
    return r.pos
end
function observation(r::ClientResults)
    return r.o
end
function read!(r::ClientResults, conn)
    #TODO: add positions
    r.s = readline(conn)
    t = split(r.s, r.flag)
    r.p1 = split(t[1])
    r.p2 = split(t[2])
    r
end


function start(server::SniperServer, pomdp::POMDP, policy::Policy)
    
    protocol = server.protocol
    map      = pomdp.map

    sn     = serverSocket(server)
    server = listen(sn)

    println("Server Ready")

    game_over = false
    p = [1, 1]
    ns = length(collect(domain(part_obs_space(pomdp))))
    b = DiscreteBelief(ns)
    rounders = [[0.0, 0.0] [-0.5, -0.5] [0.5, -0.5] [-0.5, 0.5] [0.5, 0.5]]
    params = GameParams(rounders)
    res = ClientResults()
    while true
        conn = accept(server)
        @async begin
        stepCount = 1
        try
            println("Connected")
            while true
                println("Waiting to read")
                line = readline(conn)
                println("From Client: ", line)
    
                # begin the game
                if line == protocol[:start_game]
                    println("Starting the game")
                    #read!(res, conn)
                    line = readline(conn)
                    # initialize belief
                    # parse the line here for sp and mp
                    mp, sp = parse_results(pomdp, line)
                    ba = zeros(ns)
                    if sp == [-1,-1]
                        si = pomdp.null_obs 
                        fill!(ba, 1.0/ns)
                    else
                        si = p2i(pomdp, sp)
                        ba[si] = 1.0
                    end
                    mi = p2i(pomdp, mp)
                    println(mi, " ", si, " ", b.b)
                    # finished initializing

                # update belief and send way point info
                elseif line == protocol[:next]
                    println("Sending waypoint")
                    line = readline(conn)
                    mp, sp = parse_results(pomdp, line)
                    println(mp, " ", sp)
                    println("Belief: $(b.b)")
                    if sp == [-1,-1]
                        si = pomdp.null_obs 
                    else
                        si = p2i(pomdp, sp)
                    end
                    mi = p2i(pomdp, mp)
                    a = action(policy, b, mi)
                    update_belief!(b, pomdp, mi, a, si)
                    move!(p, pomdp, mi, a)
                    waypoint = "$(p[1]/pomdp.x_size) $(p[2]/pomdp.y_size)\n"
                    write(conn, waypoint) 

                # end game
                elseif line == protocol[:end_game]
                    println("Ending the Game")
                    b = DiscreteBelief(ns)
                elseif line == protocol[:kill]
                    println("Killing the server")
                    close(conn)
                    return nothing
                else
                    println("No Simulation")
                end
            end
        catch err
            print("Connection ended with error $err")
            close(conn)
        end
        end #async
    end
end


# TODO (max): this needs to be more forgiving to wrong types/values in the input string
function parse_results(pomdp::POMDP, s::String)
    t = split(s, "split")
    mp = float(split(t[1]))
    nmp = [1,1]
    nsp = [1,1]
    println(t, " ", t[1], " ", t[2])
    println("TEST: $(split(t[2]))")
    if split(t[2])[1] == "NULL"
        println("RIGHT")
        fill!(nsp,-1)
    else
        println("WRONG")
        sp = float(split(t[2]))
        nearest_neighbor!(nsp, pomdp, sp)
    end
    nearest_neighbor!(nmp, pomdp, mp)
    # create a state type?
    return nmp, nsp
end

function nearest_neighbor!(pp::Vector{Int64}, pomdp::POMDP, p::Vector{Float64})
    # find the closest valid neighbor 
    # want four corners
    # rescale p here
    p[1] = pomdp.x_size*p[1]; p[2] = pomdp.y_size*p[2]
    rounders = [[0.0, 0.0] [-0.5, -0.5] [0.5, -0.5] [-0.5, 0.5] [0.5, 0.5]]
    pts = zeros(Int64,2,5)
    closest = Inf
    for i = 1:5
        pts[:,i] = int(p+rounders[:,i])
        d = sqrt((pts[1,i]-p[1])^2 + (pts[2,i]-p[2])^2)
        idx = p2i(pomdp, pts[:,i])
        if !in(idx, pomdp.invalid_positions) && d < closest && inbounds(pomdp.map, pts[:,i])
            pp[1:end] = pts[1:end,i]
            closest = d
        end
    end
    println("Neighbor: $p, $pp")
    pp
end

end # module
