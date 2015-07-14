module UrbanMaps

export 
    UrbanMap,
    locals,
    inbounds


using Colladas # probably don't need this
using Maps


type UrbanMap <: Map
    buildings::Vector{Building}
    nBuildings::Int
    xSize::Int
    ySize::Int
    xl::Float64
    yl::Float64
    gridSize::Int
    # for rectangular maps only
    xCenters::Vector{Float64}
    yCenters::Vector{Float64}
    xLengths::Vector{Float64}
    yLengths::Vector{Float64}

    # TODO: need to deal with normalizing the building vertices for diff x,y sizes
    function UrbanMap(coll::Collada, xSize::Int64, ySize::Int64)
        self = new()

        nBuildings = coll.nObjects
        
        self.xSize    = xSize
        self.ySize    = ySize
        self.gridSize = xSize * ySize

        self.xl = float(xSize)
        self.yl = float(ySize)

        self.nBuildings = nBuildings

        # assumes a square grid
        self.buildings = [Polygon(xSize * c.points, c.vertices) for c in coll.normalizedObjects] 

        return self
    end

    # Generates a map with n rectangular buildings
    # all input coordinates are normalized to unit square
    function UrbanMap(n::Int64, xCenters::Vector{Float64}, yCenters::Vector{Float64},
                      xLengths::Vector{Float64}, yLengths::Vector{Float64},
                      xSize::Int64, ySize::Int64)
        self = new()

        self.xSize      = xSize
        self.ySize      = ySize
        self.gridSize   = xSize * ySize
        self.nBuildings = n

        self.xl = float(xSize)
        self.yl = float(ySize)

        buildings = Array(Polygon, 0)

        # convert centers and lengths to vertices
        xm = [-0.5, 0.5, 0.5, -0.5]
        ym = [0.5, 0.5, -0.5, -0.5]
        verts = [1,2,3,4]
        for i = 1:n
            # make vertices
            xc = xCenters[i]
            yc = yCenters[i]
            xl = xLengths[i]
            yl = yLengths[i]
            points = zeros(4,3)
            for j = 1:4
                points[j,1] = xc + xm[j] * xl 
                points[j,2] = yc + ym[j] * yl 
            end
            p = Polygon(xSize * points, verts)
            push!(buildings, p)
        end
        self.buildings = buildings

        self.xCenters = xCenters
        self.yCenters = yCenters
        self.xLengths = xLengths
        self.yLengths = yLengths
        return self
    end

    function UrbanMap(t::Symbol, xSize::Int64, ySize::Int64)
        self = new()

        self.xSize = xSize
        self.ySize = ySize

        if t == :cirlce
            self.nBuildings = 1
            self.buildings = [Circular((0.5, 0.5), 0.1)]
        end

        return self
    end

end


function locals(map::UrbanMap)
    return (map.xCenters, map.yCenters, map.xLengths, map.yLengths)
end

# generates a map with rectangular buildings
function generateMaps()
    
end

function inbounds(map::UrbanMap, x::Vector{Float64})
    if 1.0 <= x[1] <= map.xl && 1.0 <= x[2] <= map.yl
        return true
    end
    return false      
end

function inbounds(map::UrbanMap, x::Vector{Int64})
    if 1 <= x[1] <= map.xSize && 1 <= x[2] <= map.ySize
        return true
    end
    return false      
end

end # module
