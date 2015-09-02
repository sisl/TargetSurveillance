module MapPlot


using UrbanMaps
using DiscreteSniper
using PyPlot
using MOMDPs

export 
    plot, 
    plot_bins,
    plot_target,
    plot_threat,
    plot_resource,
    plot_belief

function PyPlot.plot(map::UrbanMap)
    rc("text", usetex=true)
    rc("font", family="serif")
    buildings = map.buildings
    for b in buildings
        pts = b.points
        # extra point for connecting building
        n_p = size(pts,1)
        n_d = size(pts,2)
        new_points = zeros(n_p+1, n_d) 
        new_points[1:n_p,:] = pts
        new_points[end,:] = pts[1,:]
        plot(new_points[:,1], new_points[:,2], "black",lw=2.0)
        fill(new_points[:,1], new_points[:,2], fill=false, hatch="\\\\\\")
    end
    xlim([0, map.xSize])
    ylim([0, map.ySize])
end

function plot_bins(map::UrbanMap)
    xSize = map.xSize
    ySize = map.ySize
    xdiv = zeros(Int64, 2)
    ydiv = zeros(Int64, 2)
    for x = 1:xSize
        ydiv[1] = 0
        ydiv[2] = ySize
        xdiv[1] = xdiv[2] = x
        plot(xdiv, ydiv, "--", color="k", lw=0.5)
    end
    for y = 1:ySize
        xdiv[1] = 0
        xdiv[2] = xSize
        ydiv[1] = ydiv[2] = y
        plot(xdiv, ydiv, "--", color="k", lw=0.5)
    end
end

function plot_target(pomdp::SniperPOMDP)
    t = pomdp.target
    xt = t[1]; yt = t[2]
    plot(xt, yt, "*", color="k", markersize=12.5, label="Target")
end
function plot_resource(x::Float64, y::Float64)
    plot(x, y, "v", color="k", markersize=12.5, label="Agent (Blue Team)")
end
function plot_threat(x::Float64, y::Float64)
    plot(x, y, "^", color="k", markersize=12.5, label="Threat (Red Team)")
end

function plot_belief(pomdp::SniperPOMDP, b::Belief)
    xs = pomdp.x_size; ys = pomdp.y_size
    p = zeros(Int64, 2)
    bmat = zeros(xs, ys)
    for i = 1:length(b)
        i2p!(p, pomdp, i)
        bmat[p[1], p[2]] = b[i]
    end
    pcolor(bmat', cmap="binary")
    return bmat
end

end # module
