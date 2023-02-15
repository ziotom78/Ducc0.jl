# Copyright (C) 2022-2023 Max-Planck-Society, Leo A. Bianchi
# Authors: Martin Reinecke, Leo A. Bianchi

# Formatting: using JuliaFormatter; format_file("Ducc0.jl",remove_extra_newlines=true)

module Ducc0

module Support

import ducc0_jll
const libducc = ducc0_jll.libducc_julia
#libducc = "/home/martin/codes/ducc/julia/ducc_julia.so"

struct ArrayDescriptor
    shape::NTuple{10,UInt64}  # length of every axis
    stride::NTuple{10,Int64}  # stride along every axis (in elements)
    data::Ptr{Cvoid}          # pointer to the first array element
    ndim::UInt8               # number of dimensions
    dtype::UInt8              # magic values determining the data type
end

# convert data types to type codes for communication with ducc
function typecode(tp::Type)::UInt8
    if tp <: AbstractFloat
        return sizeof(tp(0)) - 1
    elseif tp <: Unsigned
        return sizeof(tp(0)) - 1 + 32
    elseif tp <: Signed
        return sizeof(tp(0)) - 1 + 16
    elseif tp == Complex{Float32}
        return typecode(Float32) + 64
    elseif tp == Complex{Float64}
        return typecode(Float64) + 64
    end
end

function desc(arr::StridedArray{T,N})::ArrayDescriptor where {T,N}
    @assert N <= 10
    ArrayDescriptor(
        NTuple{10,UInt64}(i <= N ? size(arr)[i] : 0 for i = 1:10),
        NTuple{10,Int64}(i <= N ? strides(arr)[i] : 0 for i = 1:10),
        pointer(arr),
        N,
        typecode(T),
    )
end

const Dref = Ref{ArrayDescriptor}

export libducc, desc, Dref

end  # module Support

"""
   Fast Fourier transforms and similar operations
"""
module Fft

using ..Support

function make_axes(axes, ndim)
    ax = Csize_t[axes...]::Vector{Csize_t}
    if length(unique(ax)) < length(ax)
        throw(ArgumentError("each dimension can be transformed at most once"))
    end
    if length(ax) == 0
        ax = Csize_t[(1:ndim)...]::Vector{Csize_t}  # FIXME: can this be done in an easier fashion?
    end
    return ax
end

"""
    c2c!(x, y, axes; forward, fct, nthreads)

Computes the complex Fast Fourier Transform of `x` over the requested axes and returns it in `y`.

# Arguments
- `x::StridedArray{Complex{T}}`: the input data array.
- `y::StridedArray{Complex{T}}`: the output data array.
  Must have the same dimensions as `x`.
- `axes`: the list of axes to be transformed.
  If empty, all axes are transformed.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `fct::AbstractFloat=1.0`: factor by which the result is multiplied.
  Can be used to achieve normalized transforms.
- `nthreads::Integer=1`: number of threads to use for the transform.

# Returns
a reference to y

# Notes
- `x` and `y` can refer to the same array, resulting in an in-place transform.
"""
function c2c!(
    x::StridedArray{Complex{T}},
    y::StridedArray{Complex{T}},
    axes;
    forward::Bool = true,
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{Complex{T}} where {T<:Union{Float32,Float64}}
    ax2 = make_axes(axes, ndims(x))
    size(x) == size(y) || throw(error())
    ret = ccall(
        (:fft_c2c, libducc),
        Cint,
        (Dref, Dref, Dref, Cint, Cdouble, Csize_t),
        desc(x),
        desc(y),
        desc(ax2),
        forward,
        fct,
        nthreads,
    )
    ret != 0 && throw(error())
    return y
end

"""
    c2c(x, axes; forward, fct, nthreads)

Computes the complex Fast Fourier Transform of `x` over the requested axes.

# Arguments
- `x::StridedArray{Complex{T}}`: the input data array.
- `axes`: the list of axes to be transformed.
  If empty, all axes are transformed.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `fct::AbstractFloat=1.0`: factor by which the result is multiplied.
  Can be used to achieve normalized transforms.
- `nthreads::Integer=1`: number of threads to use for the transform

# Returns
`y::StridedArray{Complex{T}}`: the result of the transform.
  Has the same dimensions as `x`
"""
function c2c(
    x::StridedArray{Complex{T}},
    axes;
    forward::Bool = true,
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{Complex{T}} where {T<:Union{Float32,Float64}}
    return c2c!(x, Array{Complex{T}}(undef, size(x)), axes, forward=forward, fct=fct, nthreads=nthreads)
end

"""
    r2c!(x, y, axes; forward, fct, nthreads)

Computes the real-to-complex Fast Fourier Transform of `x` over the requested axes and returns it in `y`.

# Arguments
- `x::StridedArray{T}`: the input data array.
- `y::StridedArray{Complex{T}}`: the output data array.
  Must have the same dimensions as `x`, except for `axes[end]`.
  If the length of that axis was `n` on input, it is `n//2+1` on output.
- `axes`: the list of axes to be transformed.
  If not set, this is assumed to be `1:ndims(x)`.
  The real-to-complex transform will be executed along `axes[end]`,
  and will be executed first.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `fct::AbstractFloat=1.0`: factor by which the result is multiplied.
  Can be used to achieve normalized transforms.
- `nthreads::Integer=1`: number of threads to use for the transform.

# Returns
a reference to y
"""
function r2c!(
    x::StridedArray{T},
    y::StridedArray{Complex{T}},
    axes;
    forward::Bool = true,
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{Complex{T}} where {T<:Union{Float32,Float64}}
    ax2 = reverse(make_axes(axes, ndims(x)))
    ret = ccall(
        (:fft_r2c, libducc),
        Cint,
        (Dref, Dref, Dref, Cint, Cdouble, Csize_t),
        desc(x),
        desc(y),
        desc(ax2),
        forward,
        fct,
        nthreads,
    )
    ret != 0 && throw(error())
    return y
end

"""
    c2r!(x, y, axes; forward, fct, nthreads)

Computes the complex-to-real Fast Fourier Transform of `x` over the requested axes and returns it in `y`.

# Arguments
- `x::StridedArray{Complex{T}}`: the input data array.
- `y::StridedArray{T}`: the output data array.
  Must have the same dimensions as `x`, except for `axes[end]`.
  If the length of that axis was `n` on input, the length of this axis must be
  either `2*n-2` or `2*n-1`.
- `axes`: the list of axes to be transformed.
  If not set, this is assumed to be `1:ndims(x)`.
  The complex-to-real transform will be executed along `axes[end]`,
  and will be executed last.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `fct::AbstractFloat=1.0`: factor by which the result is multiplied.
  Can be used to achieve normalized transforms.
- `nthreads::Integer=1`: number of threads to use for the transform.

# Returns
a reference to y
"""
function c2r!(
    x::StridedArray{Complex{T}},
    y::StridedArray{T},
    axes;
    forward::Bool = true,
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{T} where {T<:Union{Float32,Float64}}
    ax2 = reverse(make_axes(axes, ndims(x)))
    ret = ccall(
        (:fft_c2r, libducc),
        Cint,
        (Dref, Dref, Dref, Cint, Cdouble, Csize_t),
        desc(x),
        desc(y),
        desc(ax2),
        forward,
        fct,
        nthreads,
    )
    ret != 0 && throw(error())
    return y
end
function r2r_genuine_fht!(
    x::StridedArray{T},
    y::StridedArray{T},
    axes;
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{T} where {T<:Union{Float32,Float64}}
    ax2 = make_axes(axes, ndims(x))
    size(x) == size(y) || throw(error())
    ret = ccall(
        (:fft_r2r_genuine_fht, libducc),
        Cint,
        (Dref, Dref, Dref, Cdouble, Csize_t),
        desc(x),
        desc(y),
        desc(ax2),
        fct,
        nthreads,
    )
    ret != 0 && throw(error())
    return y
end

"""
    r2r_genuine_fht(x, axes; fct, nthreads)

Computes the full (non-separable) Hartley Transform of `x` over the requested axes.

# Arguments
- `x::StridedArray{T}`: the input data array.
- `axes`: the list of axes to be transformed.
  If empty, all axes are transformed.
- `fct::AbstractFloat=1.0`: factor by which the result is multiplied.
  Can be used to achieve normalized transforms.
- `nthreads::Integer=1`: number of threads to use for the transform

# Returns
`y::StridedArray{T}`: the result of the transform.
  Has the same dimensions as `x`
"""
function r2r_genuine_fht(
    x::StridedArray{T},
    axes;
    fct::AbstractFloat = 1.0,
    nthreads::Integer = 1,
)::StridedArray{T} where {T<:Union{Float32,Float64}}
    return r2r_genuine_fht!(x, Array{T}(undef, size(x)), axes, fct=fct, nthreads=nthreads)
end

end  # module Fft

"""
   Nonuniform Fast Fourier transforms
"""
module Nufft

using ..Support

"""
    best_epsilon(ndim, singleprec; sigma_min, sigma_max)

Returns the best achievable NUFFT accuracy for a given set of parameters

# Arguments
- `ndim::Integer`: the dimensionality of the transform (1-3).
- `singleprec::Bool`: whether the NUFFT is single or double precision.
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor

# Returns
AbstractFloat: the best achievable accuracy.
"""
function best_epsilon(
    ndim::Integer,
    singleprec::Bool;
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
)::Float64
    res = ccall(
        (:nufft_best_epsilon, libducc),
        Cdouble,
        (Csize_t, Cint, Cdouble, Cdouble),
        ndim,
        singleprec,
        sigma_min,
        sigma_max,
    )
    res <= 0 && throw(error())
    return res
end

"""
    u2nu!(coord, grid, points; forward, verbose, epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

Carries out a uniform-to-nonuniform (i.e. Type 2) NUFFT

# Arguments
- `coord::StridedArray{T,2}`: the coordinates of the nonuniform points. Shape `(D,npoints)`.
- `grid::StridedArray{T2,D}`: the uniform input data array.
- `points::StridedArray{T2,1}`: the output array for the non-uniform points. Shape `(npoints,)`.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.
- `epsilon::AbstractFloat=1e-5`: the requested accuracy for the transform
- `nthreads::Integer=1`: number of threads to use for the transform
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor
- `periodicity::AbstractFloat=2π`: the periodicity of the function in the nonuniform space
- `fft_order::Bool=true`: if true, assume the input array to be arranged in standard FFT order

# Returns
A reference to `points`
"""
function u2nu!(
    coord::StridedArray{T,2},
    grid::StridedArray{Complex{T2},D},
    points::StridedArray{Complex{T2},1};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Integer = 1,
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true,
)::StridedArray{Complex{T2},1} where {T<:Union{Float32,Float64},T2<:Union{Float32,Float64},D}
    GC.@preserve coord grid points
    ret = ccall(
        (:nufft_u2nu, libducc),
        Cint,
        (
            Dref,
            Dref,
            Cint,
            Cdouble,
            Csize_t,
            Dref,
            Csize_t,
            Cdouble,
            Cdouble,
            Cdouble,
            Cint,
        ),
        desc(grid),
        desc(coord),
        0,
        epsilon,
        nthreads,
        desc(points),
        verbose,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )
    ret != 0 && throw(error())
    return points
end

"""
    u2nu(coord, grid, points; forward, verbose, epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

Carries out a uniform-to-nonuniform (i.e. Type 2) NUFFT

# Arguments
- `coord::StridedArray{T,2}`: the coordinates of the nonuniform points. Shape `(D,npoints)`.
- `grid::StridedArray{T2,D}`: the uniform input data array.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.
- `epsilon::AbstractFloat=1e-5`: the requested accuracy for the transform
- `nthreads::Integer=1`: number of threads to use for the transform
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor
- `periodicity::AbstractFloat=2π`: the periodicity of the function in the nonuniform space
- `fft_order::Bool=true`: if true, assume the input array to be arranged in standard FFT order

# Returns
`StridedArray{T2,1}`: the values at the non-uniform points. Shape `(npoints,)`.
"""
function u2nu(
    coord::StridedArray{T,2},
    grid::StridedArray{Complex{T2},D};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Integer = 1,
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true,
)::Vector{Complex{T2}} where {T<:Union{Float32,Float64},T2<:Union{Float32,Float64},D}
    res = Vector{Complex{T2}}(undef, size(coord)[2])
    return u2nu!(
        coord,
        grid,
        res,
        forward = forward,
        verbose = verbose,
        epsilon = epsilon,
        nthreads = nthreads,
        sigma_min = sigma_min,
        sigma_max = sigma_max,
        periodicity = periodicity,
        fft_order = fft_order,
    )
end

"""
    nu2u!(coord, points, uniform; forward, verbose, epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

Carries out a nonuniform-to-uniform (i.e. Type 1) NUFFT

# Arguments
- `coord::StridedArray{T,2}`: the coordinates of the nonuniform points. Shape `(D,npoints)`.
- `points::StridedArray{T2,1}`: the values at the non-uniform points. `Shape (npoints,)`.
- `uniform::StridedArray{T2,D}`: the uniform output data array.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.
- `epsilon::AbstractFloat=1e-5`: the requested accuracy for the transform
- `nthreads::Integer=1`: number of threads to use for the transform
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor
- `periodicity::AbstractFloat=2π`: the periodicity of the function in the nonuniform space
- `fft_order::Bool=true`: if true, assume the input array to be arranged in standard FFT order

# Returns
A reference to `uniform`
"""
function nu2u!(
    coord::StridedArray{T,2},
    points::StridedArray{Complex{T2},1},
    uniform::StridedArray{Complex{T2},D};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Integer = 1,
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true,
)::StridedArray{Complex{T2},D} where {T<:Union{Float32,Float64},T2<:Union{Float32,Float64},D}
    GC.@preserve coord points uniform
    ret = ccall(
        (:nufft_nu2u, libducc),
        Cint,
        (
            Dref,
            Dref,
            Cint,
            Cdouble,
            Csize_t,
            Dref,
            Csize_t,
            Cdouble,
            Cdouble,
            Cdouble,
            Cint,
        ),
        desc(points),
        desc(coord),
        0,
        epsilon,
        nthreads,
        desc(uniform),
        verbose,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )
    ret != 0 && throw(error())
    return uniform
end

"""
    nu2u(coord, points, N; forward, verbose, epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

Carries out a nonuniform-to-uniform (i.e. Type 1) NUFFT

# Arguments
- `coord::StridedArray{T,2}`: the coordinates of the nonuniform points. Shape `(D,npoints)`.
- `points::StridedArray{T2,1}`: the values at the non-uniform points. Shape `(npoints,)`.
- `N::NTuple{D,Int}`: the dimensions of the output data array.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.
- `epsilon::AbstractFloat=1e-5`: the requested accuracy for the transform
- `nthreads::Integer=1`: number of threads to use for the transform
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor
- `periodicity::AbstractFloat=2π`: the periodicity of the function in the nonuniform space
- `fft_order::Bool=true`: if true, assume the input array to be arranged in standard FFT order

# Returns
`StridedArray{T2,D}`: the uniform output data array. Shape `(N,)`.
"""
function nu2u(
    coord::StridedArray{T,2},
    points::StridedArray{Complex{T2},1},
    N::NTuple{D,Int};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Integer = 1,
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true,
)::Array{Complex{T2},D} where {T<:Union{Float32,Float64},T2<:Union{Float32,Float64},D}
    res = Array{Complex{T2}}(undef, N)
    return nu2u!(
        coord,
        points,
        res,
        forward = forward,
        verbose = verbose,
        epsilon = epsilon,
        nthreads = nthreads,
        sigma_min = sigma_min,
        sigma_max = sigma_max,
        periodicity = periodicity,
        fft_order = fft_order,
    )
end

mutable struct NufftPlan
    N::Vector{UInt64}
    npoints::Int
    cplan::Ptr{Cvoid}
end

function delete_plan!(plan::NufftPlan)
    if plan.cplan != C_NULL
        ret = ccall((:nufft_delete_plan, libducc), Cint, (Ptr{Cvoid},), plan.cplan)
        ret != 0 && throw(error())
        plan.cplan = C_NULL
    end
end

"""
    make_plan(coord, N; epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

Creates a plan for subsequent Type 1 or 2 NUFFTs

# Arguments
- `coord::StridedArray{T,2}`: the coordinates of the nonuniform points. Shape `(D,npoints)`.
- `N::NTuple{D,Int}`: the dimensions of the uniform data array.
- `epsilon::AbstractFloat=1e-5`: the requested accuracy for the transform
- `nthreads::Integer=1`: number of threads to use for the transform
- `sigma_min::AbstractFloat=1.1`: the minimum allowed oversamplng factor
- `sigma_max::AbstractFloat=2.6`: the maximum allowed oversamplng factor
- `periodicity::AbstractFloat=2π`: the periodicity of the function in the nonuniform space
- `fft_order::Bool=true`: if true, assume the input array to be arranged in standard FFT order

# Returns
`NufftPlan`: the opaque plan object
"""
function make_plan(
    coord::Matrix{T},
    N::NTuple{D,Int};
    epsilon::AbstractFloat = 1e-5,
    nthreads::Integer = 1,
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true,
)::NufftPlan where {T<:Union{Float32,Float64},D}
    N2 = Vector{UInt64}(undef, D)
    for i = 1:D
        N2[i] = N[i]
    end
    GC.@preserve N2 coord
    ptr = ccall(
        (:nufft_make_plan, libducc),
        Ptr{Cvoid},
        (Cint, Dref, Dref, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint),
        false,
        desc(N2),
        desc(coord),
        epsilon,
        nthreads,
        sigma_min,
        sigma_max,
        periodicity,
        fft_order,
    )

    ptr == C_NULL && throw(error())
    p = NufftPlan(N2, size(coord)[2], ptr)
    finalizer(p -> begin
        delete_plan!(p)
    end, p)

    return p
end

"""
    nu2u_planned!(plan, points, uniform; forward, verbose)

Carries out a pre-planned nonuniform-to-uniform (i.e. Type 1) NUFFT

# Arguments
- `plan::NufftPlan`: the pre-computed plan object
- `points::StridedArray{T,1}`: the values at the non-uniform points. `Shape (npoints,)`.
- `uniform::StridedArray{T,D}`: the uniform output data array.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.

# Returns
A reference to `uniform`
"""
function nu2u_planned!(
    plan::NufftPlan,
    points::StridedArray{Complex{T},1},
    uniform::StridedArray{Complex{T}};
    forward::Bool = false,
    verbose::Bool = false,
)::StridedArray{Complex{T}} where {T<:Union{Float32,Float64}}
    GC.@preserve points uniform
    ret = ccall(
        (:nufft_nu2u_planned, libducc),
        Cint,
        (Ptr{Cvoid}, Cint, Csize_t, Dref, Dref),
        plan.cplan,
        forward,
        verbose,
        desc(points),
        desc(uniform),
    )
    ret != 0 && throw(error())
    return uniform
end

"""
    nu2u_planned(coord, points; forward, verbose)

Carries out a pre-planned nonuniform-to-uniform (i.e. Type 1) NUFFT

# Arguments
- `plan::NufftPlan`: the pre-computed plan object
- `points::StridedArray{T,1}`: the values at the non-uniform points. `Shape (npoints,)`.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.

# Returns
`StridedArray{T,D}`: the uniform output data array. Shape `(N,)`.
"""
function nu2u_planned(
    plan::NufftPlan,
    points::StridedArray{Complex{T},1};
    forward::Bool = false,
    verbose::Bool = false,
)::Array{Complex{T}} where {T<:Union{Float32,Float64}}
    res = Array{Complex{T}}(undef, Tuple(i for i in plan.N))
    nu2u_planned!(plan, points, res, forward = forward, verbose = verbose)
    return res
end

"""
    u2nu_planned!(plan, uniform, points; forward, verbose)

Carries out a pre-planned uniform-to-nonuniform (i.e. Type 2) NUFFT

# Arguments
- `plan::NufftPlan`: the pre-computed plan object
- `uniform::StridedArray{T,D}`: the uniform input data array.
- `points::StridedArray{T,1}`: the output values at the non-uniform points. `Shape (npoints,)`.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.

# Returns
A reference to `points`
"""
function u2nu_planned!(
    plan::NufftPlan,
    uniform::StridedArray{Complex{T}},
    points::StridedArray{Complex{T},1};
    forward::Bool = true,
    verbose::Bool = false,
)::StridedArray{Complex{T},1} where {T<:Union{Float32,Float64}}
    GC.@preserve uniform points
    ret = ccall(
        (:nufft_u2nu_planned, libducc),
        Cint,
        (Ptr{Cvoid}, Cint, Csize_t, Dref, Dref),
        plan.cplan,
        forward,
        verbose,
        desc(uniform),
        desc(points),
    )
    ret != 0 && throw(error())
    return points
end

"""
    u2nu_planned(plan, uniform; forward, verbose)

Carries out a pre-planned uniform-to-nonuniform (i.e. Type 2) NUFFT

# Arguments
- `plan::NufftPlan`: the pre-computed plan object
- `uniform::StridedArray{T,D}`: the uniform input data array.
- `forward::Bool=true`: if `true`, transform from real to frequency space,
  otherwise from frequency space to real space.
- `verbose::Bool=false`: if `true`, write some diagnostic output to the console.

# Returns
`StridedArray{T,1}`: the output values at the non-uniform points. `Shape (npoints,)
"""
function u2nu_planned(
    plan::NufftPlan,
    uniform::StridedArray{Complex{T}};
    forward::Bool = true,
    verbose::Bool = false,
)::Array{Complex{T},1} where {T<:Union{Float32,Float64}}
    res = Array{Complex{T}}(undef, plan.npoints)
    u2nu_planned!(plan, uniform, res, forward = forward, verbose = verbose)
    return res
end

end  # module Nufft

module Sht

using ..Support

"""
    alm2leg!(
        alm::StridedArray{Complex{T},2}, leg::StridedArray{Complex{T},3}, spin::Integer,
        lmax::Integer, mval::StridedArray{Csize_t,1}, mstart::StridedArray{Cptrdiff_t,1},
        lstride::Integer, theta::StridedArray{Cdouble,1}, nthreads::Integer = 1) where {T}

    Transforms a set of spherical harmonic coefficients to Legendre coefficients dependent on theta and m and places the result in `leg`.

# Arguments:

    - `alm::StridedArray{Complex{T},2}`: the set of spherical harmonic coefficients. ncomp must be 1 if spin is 0, else 2. The second dimension must be large enough to accommodate all entries, which are stored according to the parameters lmax, ‘mval`, mstart, and lstride.

    - `leg::StridedArray{Complex{T},3}`: output array containing the Legendre coefficients.

    - `spin::Integer`: the spin to use for the transform if spin==0, ncomp must be 1, otherwise 2.

    - `lmax::Integer`: the maximum l moment of the transform (inclusive).

    - `mval::StridedArray{Csize_t,1}`: the m moments for which the transform should be carried out, entries must be unique and <= lmax.

    - `mstart::StridedArray{Cptrdiff_t,1}`: the (hypothetical) 1-BASED index in the second dimension of alm on which the entry with l=0, m=mval[mi] would be stored, for mi in mval.

    - `lstride::Integer`: the index stride in the second dimension of alm between the entries for l and l+1, but the same m.

    - `theta::StridedArray{Cdouble,1}`: the colatitudes of the map rings.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

"""
function alm2leg!(
    alm::StridedArray{Complex{T},2},
    leg::StridedArray{Complex{T},3},
    spin::Integer,
    lmax::Integer,
    mval::StridedArray{Csize_t,1},
    mstart::StridedArray{Cptrdiff_t,1}, # 1-based
    lstride::Integer,
    theta::StridedArray{Cdouble,1},
    nthreads::Integer = 1,
)::StridedArray{Complex{T},3} where {T<:Union{Float32,Float64}}
    GC.@preserve alm mval mstart theta leg begin
        ret = ccall(
            (:sht_alm2leg, libducc),
            Cint,
            (Dref, Csize_t, Csize_t, Dref, Dref, Cptrdiff_t, Dref, Csize_t, Dref),
            desc(alm),
            spin,
            lmax,
            desc(mval),
            desc(mstart),
            lstride,
            desc(theta),
            nthreads,
            desc(leg),
        )
    end
    ret != 0 && throw(error())
    return leg
end

"""
    alm2leg(
        alm::StridedArray{Complex{T},2}, spin::Integer,
        lmax::Integer, mval::StridedArray{Csize_t,1}, mstart::StridedArray{Cptrdiff_t,1},
        lstride::Integer, theta::StridedArray{Cdouble,1}, nthreads::Integer = 1) where {T}

    Transforms a set of spherical harmonic coefficients to Legendre coefficients dependent on theta and m.

# Arguments:

    - `alm::StridedArray{Complex{T},2}`: the set of spherical harmonic coefficients. ncomp must be 1 if spin is 0, else 2. The second dimension must be large enough to accommodate all entries, which are stored according to the parameters lmax, ‘mval`, mstart, and lstride.

    - `spin::Integer`: the spin to use for the transform if spin==0, ncomp must be 1, otherwise 2.

    - `lmax::Integer`: the maximum l moment of the transform (inclusive).

    - `mval::StridedArray{Csize_t,1}`: the m moments for which the transform should be carried out, entries must be unique and <= lmax.

    - `mstart::StridedArray{Cptrdiff_t,1}`: the (hypothetical) 1-BASED index in the second dimension of alm on which the entry with l=0, m=mval[mi] would be stored, for mi in mval.

    - `lstride::Integer`: the index stride in the second dimension of alm between the entries for l and l+1, but the same m.

    - `theta::StridedArray{Cdouble,1}`: the colatitudes of the map rings.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

# Returns:

    - `leg::Array{Complex{T},3}`: output array containing the Legendre coefficients.
"""
function alm2leg(
    alm::StridedArray{Complex{T},2},
    spin::Integer,
    lmax::Integer,
    mval::StridedArray{Csize_t,1},
    mstart::StridedArray{Cptrdiff_t,1}, # 1-based
    lstride::Integer,
    theta::StridedArray{Cdouble,1},
    nthreads::Integer = 1,
)::Array{Complex{T},3} where T<:Union{Float32,Float64}
    ncomp = size(alm, 2)
    ntheta = size(theta, 1)
    nm = size(mval, 1)
    leg = Array{Complex{T}}(undef, (nm, ntheta, ncomp))
    alm2leg!(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads)
    return leg
end

"""
    leg2alm!(
        leg::StridedArray{Complex{T},3}, alm::StridedArray{Complex{T},2}, spin::Integer,
        lmax::Integer, mval::StridedArray{Csize_t,1}, mstart::StridedArray{Cptrdiff_t,1},
        lstride::Integer, theta::StridedArray{Cdouble,1}, nthreads::Integer = 1) where {T}

    Transforms a set of Legendre coefficients to spherical harmonic coefficients and places the result in `alm`.

# Arguments:

    - `leg::StridedArray{Complex{T},3}`: input array containing the Legendre coefficients. ncomp must be 1 if spin is 0, else 2. The first [m,:,:] and second [:,θ,:] dimensions must match size and ordering of `mval` and `theta` respectively.

    - `alm::StridedArray{Complex{T},2}`: the output set of spherical harmonic coefficients. ncomp must be 1 if spin is 0, else 2. The first dimension must be large enough to accommodate all entries, which are stored according to the parameters lmax, ‘mval`, mstart, and lstride.

    - `spin::Integer`: the spin to use for the transform if spin==0, ncomp must be 1, otherwise 2.

    - `lmax::Integer`: the maximum l moment of the transform (inclusive).

    - `mval::StridedArray{Csize_t,1}`: the m moments for which the transform should be carried out, entries must be unique and <= lmax.

    - `mstart::StridedArray{Cptrdiff_t,1}`: the (hypothetical) 1-BASED index in the second dimension of alm on which the entry with l=0, m=mval[mi] would be stored, for mi in mval.

    - `lstride::Integer`: the index stride in the second dimension of alm between the entries for l and l+1, but the same m.

    - `theta::StridedArray{Cdouble,1}`: the colatitudes of the map rings.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

"""
function leg2alm!(
    leg::StridedArray{Complex{T},3},
    alm::StridedArray{Complex{T},2},
    spin::Integer,
    lmax::Integer,
    mval::StridedArray{Csize_t,1},
    mstart::StridedArray{Cptrdiff_t,1}, # 1-based
    lstride::Integer,
    theta::StridedArray{Cdouble,1},
    nthreads::Integer = 1,
)::StridedArray{Complex{T},2} where {T<:Union{Float32,Float64}}
    GC.@preserve leg mval mstart theta alm begin
        ret = ccall(
            (:sht_leg2alm, libducc),
            Cint,
            (Dref, Csize_t, Csize_t, Dref, Dref, Cptrdiff_t, Dref, Csize_t, Dref),
            desc(leg),
            spin,
            lmax,
            desc(mval),
            desc(mstart),
            lstride,
            desc(theta),
            nthreads,
            desc(alm),
        )
    end
    ret != 0 && throw(error())
    return alm
end

getNalm(mstart::Cptrdiff_t, lmax::Integer, lstride::Integer) = mstart + (lmax)*lstride
getNalm(mstart::StridedArray{Cptrdiff_t,1}, lmax::Integer, lstride::Integer) =
    maximum(getNalm.(mstart, lmax, lstride))

"""
    leg2alm(
        leg::StridedArray{Complex{T},3}, spin::Integer,
        lmax::Integer, mval::StridedArray{Csize_t,1}, mstart::StridedArray{Cptrdiff_t,1},
        lstride::Integer, theta::StridedArray{Cdouble,1}, nthreads::Integer = 1) where {T}

    Transforms a set of Legendre coefficients to spherical harmonic coefficients.

# Arguments:

    - `leg::StridedArray{Complex{T},3}`: input array containing the Legendre coefficients. ncomp must be 1 if spin is 0, else 2. The first [m,:,:] and second [:,θ,:] dimensions must match size and ordering of `mval` and `theta` respectively.

    - `spin::Integer`: the spin to use for the transform if spin==0, ncomp must be 1, otherwise 2.

    - `lmax::Integer`: the maximum l moment of the transform (inclusive).

    - `mval::StridedArray{Csize_t,1}`: the m moments for which the transform should be carried out, entries must be unique and <= lmax.

    - `mstart::StridedArray{Cptrdiff_t,1}`: the (hypothetical) 1-BASED index in the second dimension of alm on which the entry with l=0, m=mval[mi] would be stored, for mi in mval.

    - `lstride::Integer`: the index stride in the second dimension of alm between the entries for l and l+1, but the same m.

    - `theta::StridedArray{Cdouble,1}`: the colatitudes of the map rings.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

# Returns:

    - `alm::StridedArray{Complex{T},2}`: the output set of spherical harmonic coefficients.
"""
function leg2alm(
    leg::StridedArray{Complex{T},3},
    spin::Integer,
    lmax::Integer,
    mval::StridedArray{Csize_t,1},
    mstart::StridedArray{Cptrdiff_t,1}, # 1-based
    lstride::Integer,
    theta::StridedArray{Cdouble,1},
    nthreads::Integer = 1,
)::Array{Complex{T},2} where {T<:Union{Float32,Float64}}
    ncomp = size(leg, 3)
    nalm = getNalm(mstart, lmax, lstride)
    alm = Array{Complex{T}}(undef, (nalm, ncomp))
    leg2alm!(leg, alm, spin, lmax, mval, mstart, lstride, theta, nthreads)
end

"""
    leg2map!(
        leg::StridedArray{Complex{T},3}, map::StridedArray{T,2}, nphi::StridedArray{Csize_t,1},
        phi0::StridedArray{Cdouble,1},ringstart::StridedArray{Csize_t,1}, pixstride::Integer,
        nthreads::Integer = 1) where {T}

    Transforms a set of Legendre coefficients to a map and places the result in `map`.

# Arguments:

    - `leg::StridedArray{Complex{T},3}`: input array containing the Legendre coefficients. The entries in leg[m,:,:] correspond to quantum number m, i.e. the m values must be stored in ascending order, and complete.

    - `map::StridedArray{T,2}`: the map pixel data. The first dimension must be large enough to accommodate all pixels, which are stored according to the parameters nphi, ‘ringstart`, and pixstride.

    - `nphi::StridedArray{Csize_t,1}`: number of pixels in every ring.

    - `phi0::StridedArray{Cdouble,1}`: azimuth (in radians) of the first pixel in every ring.

    - `ringstart::StridedArray{Csize_t,1}`: the index in the second dimension of map at which the first pixel of every ring is stored.

    - `pixstride::Integer`: the index stride in the second dimension of map between two subsequent pixels in a ring.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

"""
function leg2map!(
    leg::StridedArray{Complex{T},3},
    map::StridedArray{T,2},
    nphi::StridedArray{Csize_t,1},
    phi0::StridedArray{Cdouble,1},
    ringstart::StridedArray{Csize_t,1}, # 1-based
    pixstride::Integer,
    nthreads::Integer = 1,
)::StridedArray{T,2} where {T<:Union{Float32,Float64}}
    GC.@preserve leg nphi phi0 ringstart map begin
        ret = ccall(
            (:sht_leg2map, libducc),
            Cint,
            (Dref, Dref, Dref, Dref, Cptrdiff_t, Csize_t, Dref),
            desc(leg),
            desc(nphi),
            desc(phi0),
            desc(ringstart),
            pixstride,
            nthreads,
            desc(map),
        )
    end
    ret != 0 && throw(error())
    return map
end

getNpix(rstart::Csize_t, nphi::Csize_t, pixstride::Integer) = rstart + (nphi - 1) * pixstride
getNpix(rstart::StridedArray{Csize_t,1}, nphi::StridedArray{Csize_t,1}, pixstride::Integer) =
    maximum(getNpix.(rstart, nphi, pixstride))

"""
    leg2map(
        leg::StridedArray{Complex{T},3}, map::StridedArray{T,2}, nphi::StridedArray{Csize_t,1},
        phi0::StridedArray{Cdouble,1},ringstart::StridedArray{Csize_t,1}, pixstride::Integer,
        nthreads::Integer = 1) where {T}

    Transforms a set of Legendre coefficients to a map.

# Arguments:

    - `leg::StridedArray{Complex{T},3}`: input array containing the Legendre coefficients. The entries in leg[m,:,:] correspond to quantum number m, i.e. the m values must be stored in ascending order, and complete.

    - `nphi::StridedArray{Csize_t,1}`: number of pixels in every ring.

    - `phi0::StridedArray{Cdouble,1}`: azimuth (in radians) of the first pixel in every ring.

    - `ringstart::StridedArray{Csize_t,1}`: the index in the second dimension of map at which the first pixel of every ring is stored.

    - `pixstride::Integer`: the index stride in the second dimension of map between two subsequent pixels in a ring.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

# Returns:

    - `map::StridedArray{T,2}`: the output map pixel array.

"""
function leg2map(
    leg::StridedArray{Complex{T},3},
    nphi::StridedArray{Csize_t,1},
    phi0::StridedArray{Cdouble,1},
    ringstart::StridedArray{Csize_t,1}, # 1-based
    pixstride::Integer,
    nthreads::Integer = 1,
)::Array{T,2} where {T}
    ncomp = size(leg, 3)
    npix = getNpix(ringstart, nphi, pixstride)
    map = Array{T}(undef, (npix, ncomp))
    leg2map!(leg, map, nphi, phi0, ringstart, pixstride, nthreads)
end

"""
    map2leg!(
        map::StridedArray{T,2}, leg::StridedArray{Complex{T},3}, nphi::StridedArray{Csize_t,1},
        phi0::StridedArray{Cdouble,1},ringstart::StridedArray{Csize_t,1}, pixstride::Integer,
        nthreads::Integer = 1) where {T}

    Transforms a map to a set of Legendre coefficients dependent on theta and m, placing the result in `leg`.

# Arguments:

    - `map::StridedArray{T,2}`: the map pixel data. The first dimension must be large enough to accommodate all pixels, which are stored according to the parameters nphi, ‘ringstart`, and pixstride.

    - `leg::StridedArray{Complex{T},3}`: output array containing the Legendre coefficients. The entries in leg[m,:,:] correspond to quantum number m, i.e. the m values will be stored in ascending order, and complete.

    - `nphi::StridedArray{Csize_t,1}`: number of pixels in every ring.

    - `phi0::StridedArray{Cdouble,1}`: azimuth (in radians) of the first pixel in every ring.

    - `ringstart::StridedArray{Csize_t,1}`: the index in the second dimension of map at which the first pixel of every ring is stored.

    - `pixstride::Integer`: the index stride in the second dimension of map between two subsequent pixels in a ring.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

"""
function map2leg!(
    map::StridedArray{T,2},
    leg::StridedArray{Complex{T},3},
    nphi::StridedArray{Csize_t,1},
    phi0::StridedArray{Cdouble,1},
    ringstart::StridedArray{Csize_t,1}, # 1-based
    pixstride::Integer,
    nthreads::Integer = 1,
)::StridedArray{Complex{T},3} where T<:Union{Float32,Float64}
    GC.@preserve map nphi phi0 ringstart leg begin
        ret = ccall(
            (:sht_map2leg, libducc),
            Cint,
            (Dref, Dref, Dref, Dref, Cptrdiff_t, Csize_t, Dref),
            desc(map),
            desc(nphi),
            desc(phi0),
            desc(ringstart),
            pixstride,
            nthreads,
            desc(leg),
        )
    end
    ret != 0 && throw(error())
    return leg
end

"""
    map2leg(
        map::StridedArray{T,2}, nphi::StridedArray{Csize_t,1}, phi0::StridedArray{Cdouble,1},
        ringstart::StridedArray{Csize_t,1}, mmax::Integer, pixstride::Integer,
        nthreads::Integer = 1) where {T}

    Transforms a map to a set of Legendre coefficients dependent on theta and m.

# Arguments:

    - `map::StridedArray{T,2}`: the map pixel data. The first dimension must be large enough to accommodate all pixels, which are stored according to the parameters nphi, ‘ringstart`, and pixstride.

    - `nphi::StridedArray{Csize_t,1}`: number of pixels in every ring.

    - `phi0::StridedArray{Cdouble,1}`: azimuth (in radians) of the first pixel in every ring.

    - `ringstart::StridedArray{Csize_t,1}`: the index in the second dimension of map at which the first pixel of every ring is stored.

    - `mmax::Integer`: the maximum m moment of the transform (inclusive).

    - `pixstride::Integer`: the index stride in the second dimension of map between two subsequent pixels in a ring.

    - `nthreads::Integer = 1`: the number of threads to use for the computation if 0, use as many threads as there are hardware threads available on the system.

# Returns:

    - `leg::StridedArray{Complex{T},3}`: output array containing the Legendre coefficients. The entries in leg[m,:,:] correspond to quantum number m, i.e. the m values will be stored in ascending order, and complete.

"""
function map2leg(
    map::StridedArray{T,2},
    nphi::StridedArray{Csize_t,1},
    phi0::StridedArray{Cdouble,1},
    ringstart::StridedArray{Csize_t,1}, # 1-based
    mmax::Integer,
    pixstride::Integer,
    nthreads::Integer = 1,
)::Array{Complex{T},3} where {T<:Union{Float32,Float64}}
    ncomp = size(map, 2)
    ntheta = length(ringstart)
    leg = Array{Complex{T}}(undef, (mmax + 1, ntheta, ncomp))
    map2leg!(map, leg, nphi, phi0, ringstart, pixstride, nthreads)
end

end  # module Sht

end  # module Ducc0
