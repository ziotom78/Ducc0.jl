# proposed interface to pass array and type information from Julia to C++
# TODO: when C++ throws an exception, the whole Julia interpreter crashes ...
# would be nice to avoid this and throw a Julia exception instead.

# This code does not work out of the box since I have not updated ducc0_jll yet.
# It should mainly serve as a base for discussion.

struct ArrayDescriptor
    shape::NTuple{5,UInt64}  # length of every axis
    stride::NTuple{5,Int64}  # stride along every axis (in elements)
    data::Ptr{Cvoid}         # pointer to the first array element
    ndim::UInt8              # number of dimensions
    dtype::UInt8             # magic values determining the data type
end

function ArrayDescriptor(arr::StridedArray{T, N}) where {T,N}
    @assert N < 5
# MR the next lines just serve to put shape and stride information into the
# fixed-size tuples of the descriptor ... is tere an easier way to do this?
    shp = zeros(UInt64,5)
    str = zeros(Int64,5)
    for i in 1:N
        shp[i]=size(arr)[i]
        str[i]=strides(arr)[i]
    end
    shp = NTuple{5,UInt64}(v for v in shp)
    str = NTuple{5,Int64}(v for v in str)
# .. up to here

# MR this should probably be a static variable if such a thing exists
    typedict = Dict(Float32=>68,
                    Float64=>72,
                    Complex{Float32}=>200,
                    Complex{Float64}=>208,
                    UInt64=>40)
    tcode = typedict[T]
    ArrayDescriptor(shp, str, pointer(arr), N, tcode,)
end

# This is the function that should be called by the end user
function ducc_u2nu(coord::StridedArray{Float64,2}, grid::StridedArray{T,N};
    forward::Bool = true,
    verbose::Bool = false,
    epsilon::AbstractFloat = 1e-5,
    nthreads::Unsigned = UInt32(1),
    sigma_min::AbstractFloat = 1.1,
    sigma_max::AbstractFloat = 2.6,
    periodicity::AbstractFloat = 2π,
    fft_order::Bool = true) where {T,N}

    res = Vector{T}(undef,size(coord)[2])
# MR which variables do I need to preserve?
    GC.@preserve coord grid res ccall((:nufft_u2nu_julia_double_new, "./ducc_julia.so"), Cvoid, (ArrayDescriptor,ArrayDescriptor, Cint, Cdouble, Csize_t, ArrayDescriptor,Csize_t, Cdouble, Cdouble, Cdouble, Cint), ArrayDescriptor(grid), ArrayDescriptor(coord), 0, epsilon, nthreads, ArrayDescriptor(res), verbose, sigma_min, sigma_max, periodicity, fft_order)
    return res
end

mutable struct NufftPlan
  N::Vector{UInt64}
  npoints::Int
  cplan::Ptr{Cvoid}
end

function NufftPlan(coords::Matrix{T}, N::NTuple{D,Int}; nu2u::Bool=false,
                             epsilon::AbstractFloat=1e-5,
                             nthreads::Unsigned = UInt32(1),
                             sigma_min::AbstractFloat=1.1,
                             sigma_max::AbstractFloat=2.6,
                             periodicity::AbstractFloat = 2π,
                             fft_order::Bool = true) where {T,D}
  N2=Vector{UInt64}(undef,D)
  for i in 1:D
    N2[i] = N[i]
  end
  ptr = ccall((:make_nufft_plan_julia, "./ducc_julia.so"), Ptr{Cvoid}, 
                (Cint, ArrayDescriptor, ArrayDescriptor, Cdouble, Csize_t, Cdouble, Cdouble, Cdouble, Cint), 
                nu2u, ArrayDescriptor(N2), ArrayDescriptor(coords), epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order)

  p = NufftPlan(N2, size(coords)[2], ptr)

  finalizer(p -> begin
    println("finalize!")
    ccall((:delete_nufft_plan_julia, "./ducc_julia.so" ), Cvoid, (Ptr{Cvoid},), p.cplan)
  end, p)

  return p
end
function planned_nu2u(plan::NufftPlan, points::StridedArray{T,1}; forward::Bool=true, verbose::Bool=false,) where {T}
  res = Array{T}(undef, Tuple(i for i in plan.N))
  ccall((:planned_nu2u_julia, "./ducc_julia.so" ), Cvoid, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(points), ArrayDescriptor(res))
  return res
end
function planned_u2nu(plan::NufftPlan, uniform::StridedArray{T}; forward::Bool=true, verbose::Bool=false,) where {T}
  res = Array{T}(undef, plan.npoints)
  ccall((:planned_u2nu_julia, "./ducc_julia.so" ), Cvoid, (Ptr{Cvoid}, Cint, Csize_t, ArrayDescriptor, ArrayDescriptor), plan.cplan, forward, verbose, ArrayDescriptor(uniform), ArrayDescriptor(res))
  return res
end


# demo call
npoints=1000000
shp=(1000,1000)
coord = rand(Float64, length(shp),npoints) .- Float32(0.5)
plan = NufftPlan(coord, shp)
points = rand(Complex{Float64},(npoints,))
planned_nu2u(plan, points)
grid = ones(Complex{Float64},shp)
planned_u2nu(plan, grid)
points = rand(Complex{Float32},(npoints,))
planned_nu2u(plan, points)
grid = ones(Complex{Float32},shp)
planned_u2nu(plan, grid)
