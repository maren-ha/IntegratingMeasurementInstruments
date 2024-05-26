#------------------------------
# ODE solutions
#------------------------------

#analyticalsol(t, p, x0::Float32) = (x0 .+ (p[2]./p[1])).*exp.(p[1].*t) .- (p[2]./p[1]) #1dim only

#------------------------------
# calculate analytical solution from A, c, x0
#------------------------------


function generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32})
    eAt = exp(A.*t)
    return eAt*(c + x0) - c, eAt
end

function generalsolution(t, x0::Vector{Float32}, p::Vector{Float32}) # for drift only solution 
    return p.*t + x0, 1.0f0
end

#------------------------------
# get parameters for each system 
#------------------------------

function params_fullinhomogeneous(p::Vector{Float32})
    if length(p) != 6
        error("2D inhomogeneous linear system requires 6 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    c = inv(A)*p[5:6]
    return A, c
end

function params_offdiagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only off-diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([0.0f0 p[1] p[2] 0.0f0], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

function params_diagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

function params_driftonly(p::Vector{Float32})
    if length(p) != 2
        error("drift only solution requires 2 parameters, but p is of length $(length(p))")
    end
    return [p]
end

function params_fullhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D homogeneous linear system requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    return A, zeros(Float32,2)
end

function params_diagonalhomogeneous(p::Vector{Float32})
    if length(p) != 2
        error("2D homogeneous linear system without interactions requires 2 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    return A, zeros(Float32,2)
end