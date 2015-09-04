# interface for exporting Collada (.dae) files for julia
module Colladas

export Collada,
       ColladaObjects,
       getObjectData

using LightXML


abstract Collada


# MapObjects - contains points and vertices of object in .dae file
type MapObject

    points::Matrix

    vertices::Vector

end



# collada type - contains objects in .dae file
type ColladaObjects <: Collada 

    name::ASCIIString

    xdoc::XMLDocument

    xroot::XMLElement

    objects::Vector{MapObject}

    normalizedObjects::Vector{MapObject} # positions are normalized

    norm::Float64
    offx::Float64
    offy::Float64

    nObjects::Int

    function ColladaObjects(fileName::ASCIIString)
        self = new()
        self.name = fileName
        self.xdoc = parse_file(fileName)
        self.xroot = root(self.xdoc)
        self.objects = getObjectData(self.xroot)
        (nobj, no, ox, oy) = normalizeObjects(self.objects)
        self.normalizedObjects = nobj
        self.norm = no
        self.offx = ox
        self.offy = oy
        self.nObjects = length(self.normalizedObjects)
        return self
    end

end

function getObjectData(xroot::XMLElement)
    ces = get_elements_by_tagname(xroot, "library_geometries")
    e1 = ces[1]
    meshes = collect(child_elements(e1)) # collection of map objects
    nObjects = length(meshes) 
    objects = Array(MapObject, nObjects)
    for i = 1:nObjects
        pos, verts = getObjectProperties(meshes, i)
        objects[i] = MapObject(pos, verts)
    end
    return objects
end


function normalizeObjects(objects::Vector)
    nObjects = length(objects)
    normObjects = Array(MapObject, nObjects - 1) # one object is border
    # assume square map for simplicty
    # can incorporate x and y norms later
    normConstant = 0.0
    offx = 0.0; offy = 0.0
    zMin = Inf
    borderIdx = 0
    # assume map border is at -z plane
    for i = 1:nObjects
        currentMin = minimum(objects[i].points[:,3])
        if currentMin < zMin
            zMin = currentMin
            normConstant = maximum(objects[i].points[:,1]) - minimum(objects[i].points[:,1])
            offx = minimum(objects[i].points[:,1])
            offy = minimum(objects[i].points[:,2])
            borderIdx = i
        end
    end
    # renormalize the object points
    objCount = 1
    for i = 1:nObjects
        if i != borderIdx
            newPoints = zeros(size(objects[i].points))
            newVerts = deepcopy(objects[i].vertices)
            newPoints[:,1] = (objects[i].points[:,1] - offx) / normConstant
            newPoints[:,2] = (objects[i].points[:,2] - offy) / normConstant
            newPoints[:,3] = objects[i].points[:,3]
            #newPoints = objects[i].points / normConstant
            normObjects[objCount] = MapObject(newPoints, newVerts)
            objCount += 1
        end
    end
    return (normObjects, normConstant, offx, offy)
end


function getObjectProperties(meshes::Vector, objNum::Int)

    objMesh = get_elements_by_tagname(meshes[objNum], "mesh")[1] # mesh has a single tag object

    source = get_elements_by_tagname(objMesh, "source") # norm and pos geometry
    vertices = get_elements_by_tagname(objMesh, "vertices")[1]
    triangles = get_elements_by_tagname(objMesh, "polylist")
    # hack to work with both keys
    if isempty(triangles)
        triangles = get_elements_by_tagname(objMesh, "triangles")
    end
    triangles = triangles[1]

    inp = find_element(vertices, "input") # first input attribute
    sem = attribute(inp, "semantic")

    posIdx = 0
    if sem == "POSITION"
        posIdx = 1
    else
        posIdx = 2
    end

    rawPos = float(split(content(get_elements_by_tagname(source[posIdx], "float_array")[1])))
    vertIdx = int(split(content(get_elements_by_tagname(triangles, "p")[1]))) + 1

    pos = makePoints(rawPos)

    return pos, vertIdx

end


function makePoints(rawPts; stride=3)
    nPoints = int(length(rawPts) / stride)
    pos = zeros(nPoints, stride)
    for i = 1:nPoints
        sIdx = i*stride - (stride - 1)
        eIdx = sIdx + (stride - 1)
        pos[i,:] = rawPts[sIdx:eIdx]
    end
    return pos
end


end #module

