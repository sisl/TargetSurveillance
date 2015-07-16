# interface for client-server communication 
# used for sending simulation results
module SimServers

# interface with the simulator
using Simulator
using Map_ 

export SimServer, SniperServer
export simAndSend, send


abstract SimServer


type SniperServer <: SimServer
    socketNumber::Int64 
    sendDelay::Int64 # in ms
    protocol::Dict{Symbol, String} # desired operation to protocol
end

function SniperServer(socket::Int64; delay::Int64=0, protocol=Dict())
    if isempty(protocol)
        protocol[:SendSim]  = "0\n"
        protocol[:SendMap]  = "1\n"
        protocol[:SendSize] = "2\n"
        protocol[:End]      = "-1\n"
    end
    return SniperServer(socket, delay, protocol)
end

function serverSocket(server::SniperServer)
    return server.socketNumber
end


# decision process should be a more general abstract type?
# can be mdp or pomdp
# mdp is a suptype of pomdp
function simAndSend(server::SniperServer, sim::Simulation, mdp, policy)
    
    protocol = server.protocol
    map      = mdp.map

    sn     = serverSocket(server)
    server = listen(sn)

    println("Server Ready")

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
    
                # send simulation results for one time-step
                if line == protocol[:SendSim]    
                    println("Sending Simulation Results")
                    simulateStep!(sim, mdp, stepCount, policy)
                    send(conn, sim, stepCount)
                    stepCount += 1

                # send the map information
                elseif line == protocol[:SendMap]
                    println("Sending Map Information")
                    send(conn, map)

                # send the map size
                elseif line == protocol[:SendSize]
                    println("Sending grid size")
                    send(conn, map.xSize)

                # end simulation
                elseif line == protocol[:End]
                    # end sim 
                    return
                # nothing
                else
                    println("No Simulation")
                end
            end
        catch err
            print("connection ended with error $err")
        end
        end #async
    end
end

# sends the simulation coordinates
function send(conn, simResults::MDPSimulation, step::Int64)
    results = string(simResults.coordinates[step,:], "\n")
    write(conn, results) 
    return
end


# sends the simulation coordinates + belief
function send(conn, simResults::POMDPSimulation, step::Int64)
    results = string(int(simResults.coordinates[step,:]), "splitFlag", simResults.belief[step,:], "\n")
    write(conn, results) 
    return
end


# sends the map coordinates
function send(conn, map::Map)
    bString = ""
    nBuildings = length(map.buildings)
    count = 1
    for b in map.buildings
        # use "buildingSplit"
        # use "vertexSplit"
        verts = b.points
        nVert = size(verts, 1)
        for i = 1:nVert
            if i == nVert
                vertString = string(verts[i,:])
                bString = string(bString, vertString)
            else
                vertString = string(verts[i,:])
                bString = string(bString, vertString, "vertexSplit")
            end
        end
        if count != nBuildings
            bString = string(bString, "buildingSplit")
        end
        count += 1
    end
    bString = string(bString, "\n")
    write(conn, bString)
    return
end


# for sending grid size
function send(conn, i::Int64)
    s = string(i, "\n")
    write(conn, s)
    return
end

end # module
