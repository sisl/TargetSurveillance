module UrbanMaps

export 
    UrbanMap,
    locals,
    inbounds,
    shift!


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
    function UrbanMap(coll::Collada, xSize::Int64, ySize::Int64; 
                      shift::Bool=true, shift_val::Float64=0.0)
        self = new()

        nBuildings = coll.nObjects
        
        self.xSize    = xSize
        self.ySize    = ySize
        self.gridSize = xSize * ySize

        self.xl = float(xSize)
        self.yl = float(ySize)

        self.nBuildings = nBuildings

        # assumes a square grid
        buildings = None
        if shift
            #self.buildings = [Polygon(shift_map(xSize * (xSize-1)/xSize * c.points, xSize, ySize, shift_val), c.vertices) for c in coll.normalizedObjects]
            self.buildings = [Polygon(shift_map(xSize * c.points, xSize, ySize, shift_val), c.vertices) for c in coll.normalizedObjects]
        else
            self.buildings = [Polygon(xSize * c.points, c.vertices) for c in coll.normalizedObjects]
        end

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


function Base.shift!(map::UrbanMap, xshift::Float64, yshift::Float64)
    for i = 1:map.nBuildings
        b = map.buildings[i].points
        (npts, ndim) = size(b)
        for j = 1:npts
            b[j,1] += xshift
            b[j,2] += yshift
        end
    end
    map
end

# shift map down and left for symmetry
function shift_map(pts::Matrix{Float64}, xs::Int64, ys::Int64, shift_val::Float64)
    (np, ndim) = size(pts)
    npts = zeros(np, ndim)
    for i = 1:np
        npts[i,1] = pts[i,1] + shift_val/(xs)*xs
        npts[i,2] = pts[i,2] + shift_val/(ys)*xs
        npts[i,3] = pts[i,3] 
    end
    return npts
end

function order_points(pts::Matrix{Float64}, verts::Vector{Int64})
    (np, ndim) = size(pts)
    nv = length(verts)
    println("$nv, $np")
    @assert nv == np "Vertex and point size mismatch in .dae file"
    npts = zeros(np, ndim)
    for i = 1:np
        npts[i,:] = pts[verts[i],:]
    end
    return npts
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
