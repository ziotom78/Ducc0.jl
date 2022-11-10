module Ducc0

using Libdl
using ducc0_jll

export nufft_u2nu!, nufft_nu2u!
export NufftPlan, make_nufft_plan, delete_nufft_plan
export planned_nu2u, planned_u2nu

function nufft_u2nu!(
    ndim::Integer,
    npoints::Integer,
    shape,
    grid::Array{Float64},
    coord::Array{Float64},
    out::Array{Float64},
    sigma_min,
    sigma_max,
    fft_order;
    forward = true,
    verbosity = false,
    epsilon = 1e-7,
    nthreads = 1,
    periodicity = 2π
    )

    @assert length(coord) == ndim * npoints
    @assert length(out) == 2npoints
    
    ccall(
        (:nufft_u2nu_julia_double, libducc_julia),
        Cvoid,
        (
            Csize_t,       # ndim
            Csize_t,       # npoints
            Ptr{Csize_t},  # shape
            Ptr{Cdouble},  # grid
            Ptr{Cdouble},  # coord
            Cint,          # forward
            Cdouble,       # epsilon
            Csize_t,       # nthreads
            Ptr{Cdouble},  # out
            Csize_t,       # verbosity
            Cdouble,       # sigma_min
            Cdouble,       # sigma_max
            Cdouble,       # periodicity
            Cint,          # fft_order
        ),
        ndim,
        npoints,
        shape,
        grid,
        coord,
        forward ? 1 : 0,
        epsilon,
        nthreads,
        out,
        verbosity ? 1 : 0,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )
end

function nufft_nu2u!(
    ndim::Integer,
    npoints::Integer,
    shape,
    points::Array{Float64},
    coord::Array{Float64},
    out::Array{Float64},
    sigma_min,
    sigma_max,
    fft_order;
    forward = true,
    verbosity = false,
    epsilon = 1e-7,
    nthreads = 1,
    periodicity = 2π
    )

    @assert length(coord) == ndim * npoints
    @assert length(out) == 2prod(shape)
    
    ccall(
        (:nufft_nu2u_julia_double, libducc_julia),
        Cvoid,
        (
            Csize_t,       # ndim
            Csize_t,       # npoints
            Ptr{Csize_t},  # shape
            Ptr{Cdouble},  # points
            Ptr{Cdouble},  # coord
            Cint,          # forward
            Cdouble,       # epsilon
            Csize_t,       # nthreads
            Ptr{Cdouble},  # out
            Csize_t,       # verbosity
            Cdouble,       # sigma_min
            Cdouble,       # sigma_max
            Cdouble,       # periodicity
            Cint,          # fft_order
        ),
        ndim,
        npoints,
        shape,
        points,
        coord,
        forward ? 1 : 0,
        epsilon,
        nthreads,
        out,
        verbosity ? 1 : 0,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )
end


@doc raw"""
    struct NufftPlan

A plan for Nufft. Create one using [`make_nufft_plan`](@ref) and destroy
it using [`delete_nufft_plan`](@ref).
"""
struct NufftPlan
    ptr::Ptr{Cvoid}
end


function make_nufft_plan(
    nu2u,
    ndim,
    npoints,
    shape,
    coord::Array{Float64},
    sigma_min,
    sigma_max,
    fft_order;
    epsilon = 1e-7,
    nthreads = 1,
    periodicity = 2π
    )

    result = ccall(
        (:make_nufft_plan_double, libducc_julia),
        Ptr{Cvoid},
        (
            Cint,         # nu2u
            Csize_t,      # ndim
            Csize_t,      # npoints
            Ptr{Csize_t}, # shape
            Ptr{Cdouble}, # coord
            Cdouble,      # epsilon
            Csize_t,      # nthreads
            Cdouble,      # sigma_min
            Cdouble,      # sigma_max
            Cdouble,      # periodicity
            Cint,         # fft_order
        ),
        nu2u,
        ndim,
        npoints,
        shape,
        coord,
        epsilon,
        nthreads,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )

    return NufftPlan(result)
end

function delete_nufft_plan(plan::NufftPlan)
    ccall(
        (:delete_nufft_plan_double, libducc_julia),
        Cvoid,
        (Ptr{Cvoid},),
        plan.ptr,
    )
end

function planned_nu2u(
    plan::NufftPlan,
    points::Array{Float64},
    uniform::Array{Float64};
    forward = true,
    verbosity = false,
    )
    
    ccall(
        (:planned_nu2u, libducc_julia),
        Cvoid,
        (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
        plan.ptr,
        forward ? 1 : 0,
        verbosity ? 1 : 0,
        points,
        uniform,
    )
end

function planned_u2nu(
    plan::NufftPlan,
    points::Array{Float64},
    uniform::Array{Float64};
    forward = true,
    verbosity = false,
    )
    
    ccall(
        (:planned_u2nu, libducc_julia),
        Cvoid,
        (Ptr{Cvoid}, Cint, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
        plan.ptr,
        forward ? 1 : 0,
        verbosity ? 1 : 0,
        points,
        uniform,
    )
end


################################################################################
# Docstrings

@doc raw"""
    nufft_u2nu!(ndim, npoints, shape, grid, coord, out, sigma_min, sigma_max, fft_order; forward=true, verbosity=false, epsilon=1e-7, nthreads=1, periodicity=2π)
"""
nufft_u2nu!

@doc raw"""
    nufft_nu2u!(ndim, npoints, shape, points, coord, out, sigma_min, sigma_max, fft_order; forward=true, verbosity=false, epsilon=1e-7, nthreads=1, periodicity=2π)
"""
nufft_nu2u!

@doc raw"""
    make_nufft_plan(nu2u, ndim, npoints, shape, coord, sigma_min, sigma_max, fft_order; epsilon=1e-7, nthreads=1, periodicity=2π)
"""
make_nufft_plan

@doc raw"""
    delete_nufft_plan(plan)
"""
delete_nufft_plan

@doc raw"""
    planned_nu2u(plan, points, uniform; forward=true, verbosity=false)
"""
planned_nu2u

@doc raw"""
    planned_u2nu(plan, points, uniform; forward=true, verbosity=false)
"""
planned_u2nu

end
