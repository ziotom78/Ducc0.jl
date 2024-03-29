include("./Ducc0.jl")

module AbstractInterfaces

import ..Ducc0
import LinearAlgebra
import AbstractFFTs
using AbstractNFFTs

mutable struct Ducc0FFTPlan{T,N} <: AbstractFFTs.Plan{T}
    region::Any
    sz::NTuple{N,Int}
    forward::Bool
    normalize::Bool
    nthreads::Csize_t
    pinv::AbstractFFTs.Plan{T}
    function Ducc0FFTPlan{T}(
        region,
        sz::NTuple{N,Int},
        forward::Bool = true,
        normalize::Bool = false,
        nthreads::Int=1,
    ) where {T,N}
        return new{T,N}(region, sz, forward, normalize, nthreads)
    end
end

Base.size(p::Ducc0FFTPlan) = p.sz
Base.ndims(::Ducc0FFTPlan{T,N}) where {T,N} = N

function AbstractFFTs.plan_fft(x::AbstractArray{T}, region; kwargs...) where {T}
    return Ducc0FFTPlan{T}(region, size(x), true, false)
end
function AbstractFFTs.plan_bfft(x::AbstractArray{T}, region; kwargs...) where {T}
    return Ducc0FFTPlan{T}(region, size(x), false, false)
end

function AbstractFFTs.plan_inv(p::Ducc0FFTPlan{T}) where {T}
    res = Ducc0FFTPlan{T}(p.region, p.sz, !p.forward, !p.normalize)
    res.pinv = p
    return res
end

function mul!(
    y::AbstractArray{<:Complex,N},
    p::Ducc0FFTPlan,
    x::AbstractArray{<:Union{Complex,AbstractFloat},N},
) where {N}
    size(y) == size(p) == size(x) || throw(DimensionMismatch())
    fct = p.normalize ? AbstractFFTs.normalization(Float64, p.sz, p.region) : 1.0
    Ducc0.Fft.c2c!(x, y, p.region, forward = p.forward, fct = fct, nthreads = p.nthreads)
end

Base.:*(p::Ducc0FFTPlan, x::AbstractArray) =
    mul!(similar(x, complex(float(eltype(x)))), p, x)

mutable struct Ducc0RFFTPlan{T,N} <: AbstractFFTs.Plan{T}
    region
    sz::NTuple{N,Int}
    forward::Bool
    normalize::Bool
    nthreads::Csize_t
    pinv::AbstractFFTs.Plan{Complex{T}}
    function Ducc0RFFTPlan{T}(
        region,
        sz::NTuple{N,Int},
        forward::Bool = true,
        normalize::Bool = false,
        nthreads::Int=1,
    ) where {T,N}
        return new{T,N}(region, sz, forward, normalize, nthreads)
    end
end

mutable struct InverseDucc0RFFTPlan{T,N} <: AbstractFFTs.Plan{Complex{T}}
    d::Int
    region
    sz::NTuple{N,Int}
    forward::Bool
    normalize::Bool
    nthreads::Csize_t
    pinv::AbstractFFTs.Plan{T}
    function InverseDucc0RFFTPlan{T}(
        d::Int,
        region,
        sz::NTuple{N,Int},
        forward::Bool = false,
        normalize::Bool = false,
        nthreads::Int=1,
    ) where {T,N}
        return new{T,N}(d, region, sz, forward, normalize, nthreads)
    end
end

function AbstractFFTs.plan_rfft(x::AbstractArray{T}, region; kwargs...) where {T<:Real}
    return Ducc0RFFTPlan{T}(region, size(x))
end
function AbstractFFTs.plan_brfft(x::AbstractArray{Complex{T}}, d, region; kwargs...) where {T}
    return InverseDucc0RFFTPlan{T}(d, region, size(x))
end
function AbstractFFTs.plan_inv(p::Ducc0RFFTPlan{T,N}) where {T,N}
    firstdim = first(p.region)::Int
    d = p.sz[firstdim]
    sz = ntuple(i -> i == firstdim ? d ÷ 2 + 1 : p.sz[i], Val(N))
    res = InverseDucc0RFFTPlan{T}(d, p.region, sz, !p.forward, !p.normalize)
    res.pinv = p
    return res
end

function AbstractFFTs.plan_inv(pinv::InverseDucc0RFFTPlan{T,N}) where {T,N}
    firstdim = first(pinv.region)::Int
    sz = ntuple(i -> i == firstdim ? pinv.d : pinv.sz[i], Val(N))
    res = Ducc0RFFTPlan{T}(pinv.region, sz, !pinv.forward, !pinv.normalize)
    res.pinv = pinv
    return res
end

Base.size(p::Ducc0RFFTPlan) = p.sz
Base.ndims(::Ducc0RFFTPlan{T,N}) where {T,N} = N
Base.size(p::InverseDucc0RFFTPlan) = p.sz
Base.ndims(::InverseDucc0RFFTPlan{T,N}) where {T,N} = N

to_real!(x::AbstractArray) = map!(real, x, x)

function Base.:*(p::Ducc0RFFTPlan, x::AbstractArray)
    size(p) == size(x) || error("array and plan are not consistent")

    # create output array
    firstdim = first(p.region)::Int
    d = size(x, firstdim)
    firstdim_size = d ÷ 2 + 1
    T = complex(float(eltype(x)))
    sz = ntuple(i -> i == firstdim ? firstdim_size : size(x, i), Val(ndims(x)))
    y = similar(x, T, sz)

    fct = p.normalize ? AbstractFFTs.normalization(Float64, p.sz, p.region) : 1.0
    Ducc0.Fft.r2c!(x, y, p.region, forward = p.forward, fct = fct, nthreads = p.nthreads)
    return y
end

function Base.:*(p::InverseDucc0RFFTPlan, x::AbstractArray)
    size(p) == size(x) || error("array and plan are not consistent")

    # create output array
    firstdim = first(p.region)::Int
    d = p.d
    sz = ntuple(i -> i == firstdim ? d : size(x, i), Val(ndims(x)))
    y = similar(x, real(float(eltype(x))), sz)

    # compute DFT
    fct = p.normalize ? AbstractFFTs.normalization(Float64, sz, p.region) : 1.0
    Ducc0.Fft.c2r!(x, y, p.region, forward = p.forward, fct = fct, nthreads = p.nthreads)
    return y
end


mutable struct Ducc0NufftPlan{T,D} <: AbstractNFFTPlan{T,D,1}
    N::NTuple{D,Int64}
    J::Int64
    plan::Ducc0.Nufft.NufftPlan
end

function Ducc0NufftPlan(
    k::Matrix{T},
    N::NTuple{D,Int},
    reltol::AbstractFloat,
    nthreads::Int = 1,
) where {D,T<:Float64}
    J = size(k, 2)
    sigma_min = 1.1
    sigma_max = 2.6

    reltol = max(
        reltol,
        1.1 * Ducc0.Nufft.best_epsilon(D, false, sigma_min = sigma_min, sigma_max = sigma_max),
    )

    plan = Ducc0.Nufft.make_plan(
        k,
        N,
        epsilon = reltol,
        nthreads = nthreads,
        sigma_min = sigma_min,
        sigma_max = sigma_max,
        periodicity = 1.0,
        fft_order = false,
    )
    return Ducc0NufftPlan{T,D}(N, J, plan)
end

function Base.show(io::IO, p::Ducc0NufftPlan)
    print(io, "Ducc0NufftPlan")
end

AbstractNFFTs.size_in(p::Ducc0NufftPlan) = Int.(p.N)
AbstractNFFTs.size_out(p::Ducc0NufftPlan) = (Int(p.J),)

function AbstractNFFTs.plan_nfft(
    ::Type{<:Array},
    k::Matrix{T},
    N::NTuple{D,Int},
    rest...;
    kargs...,
) where {T,D}
    return Ducc0NufftPlan(k, N, rest...; kargs...)
end

function LinearAlgebra.mul!(
    fHat::Vector{Complex{T}},
    p::Ducc0NufftPlan{T,D},
    f::Array{Complex{T},D};
    verbose = false,
) where {T<:Float64,D}
    Ducc0.Nufft.u2nu_planned!(p.plan, f, fHat, forward = true, verbose = verbose)
    return fHat
end

function LinearAlgebra.mul!(
    f::Array{Complex{T},D},
    pl::AbstractNFFTs.Adjoint{Complex{T},<:Ducc0NufftPlan{T,D}},
    fHat::Vector{Complex{T}};
    verbose = false,
) where {T<:Float64,D}
    p = pl.parent

    Ducc0.Nufft.nu2u_planned!(p.plan, fHat, f, forward = false, verbose = verbose)
    return f
end

end
