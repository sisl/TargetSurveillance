module MapPlot


using UrbanMaps
using PyPlot

export plot

function PyPlot.plot(map::UrbanMap)
    buildings = map.buildings
    for b in buildings
        pts = b.points
        # extra point for connecting building
        n_p = size(pts,1)
        n_d = size(pts,2)
        new_points = zeros(n_p+1, n_d) 
        new_points[1:n_p,:] = pts
        new_points[end,:] = pts[1,:]
        plot(new_points[:,1], new_points[:,2] ,"black",lw=2.0)
    end
end

end # module
