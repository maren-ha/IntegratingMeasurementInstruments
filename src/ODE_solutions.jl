#------------------------------
# ODE solutions
#------------------------------

# calculate analytical solution from A, c, x0

"""
    generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32}) 

Calculates the analytical solution of a linear system of ODEs with constant coefficients 
    at a time point `t`, for a system matrix `A`, a constant vector `c` and an initial value `x0`.
    
Returns the solution `x(t)` and the matrix exponential `e^{At}`.
"""
function generalsolution(t, x0::Vector{Float32}, A::Matrix{Float32}, c::Vector{Float32})
    eAt = exp(A.*t)
    return eAt*(c + x0) - c, eAt
end

"""
    generalsolution(t, x0::Vector{Float32}, c::Vector{Float32}) 

Calculates the analytical solution of a constant system of ODEs at a time point `t`, 
    for a constant vector `c` and an initial value `x0`.
    
Returns the solution `x(t)` and the matrix exponential `e^{At}` = 1.0.
"""
function generalsolution(t, x0::Vector{Float32}, p::Vector{Float32}) # for drift only solution 
    return p.*t + x0, 1.0f0
end

# get parameters for each system 

"""
    params_fullinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system 
    with 6 parameters specified by the input vector `p` = [a11, a12, a21, a22, c1, c2].
"""
function params_fullinhomogeneous(p::Vector{Float32})
    if length(p) != 6
        error("2D inhomogeneous linear system requires 6 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    c = inv(A)*p[5:6]
    return A, c
end

"""
    params_offdiagonalinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system where 
    the system matrix has only off-diagonal elements (i.e., diagonal entries are zero), 
    specified by the input vector `p` = [a12, a21, c1, c2].
"""
function params_offdiagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only off-diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([0.0f0 p[1] p[2] 0.0f0], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

"""
    params_diagonalinhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D inhomogeneous linear system where
    the system matrix is diagonal, specified by the input vector `p` = [a11, a22, c1, c2].
"""
function params_diagonalinhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D inhomogeneous linear system with only diagonals requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    c = inv(A)*p[3:4]
    return A, c
end

"""
    params_driftonly(p::Vector{Float32})

Returns the constant vector `c` of a 2D inhomogeneous linear system where
    the system matrix is zero, specified by the input vector `p` = [c1, c2].
"""
function params_driftonly(p::Vector{Float32})
    if length(p) != 2
        error("drift only solution requires 2 parameters, but p is of length $(length(p))")
    end
    return [p]
end

"""
    params_fullhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D homogeneous linear system
    with 4 parameters specified by the input vector `p` = [a11, a12, a21, a22], and `c` = [0, 0].
"""
function params_fullhomogeneous(p::Vector{Float32})
    if length(p) != 4
        error("2D homogeneous linear system requires 4 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape(p[1:4], (2,2)))
    return A, zeros(Float32,2)
end

"""
    params_diagonalhomogeneous(p::Vector{Float32})

Returns the system matrix `A` and the constant vector `c` of a 2D homogeneous linear system where
    the system matrix is diagonal, specified by the input vector `p` = [a11, a22], and `c` = [0, 0].
"""
function params_diagonalhomogeneous(p::Vector{Float32})
    if length(p) != 2
        error("2D homogeneous linear system without interactions requires 2 parameters, but p is of length $(length(p))")
    end
    A = Matrix(reshape([p[1] 0.0f0 0.0f0 p[2]], (2,2)))
    return A, zeros(Float32,2)
end