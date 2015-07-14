module Maps


export Map, Building
export Circular, Polygon


abstract Map
abstract Building

immutable Circular <: Building
    center::(Float64, Float64)
    radius::Float64
end

immutable Polygon <: Building
    points::Matrix    # Nx3 matrix representing building points
    vertices::Vector  # vertices representing the point order
end

end # module
