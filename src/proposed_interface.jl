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
                    Complex{Float64}=>208)
    tcode = typedict[T]
    ArrayDescriptor(shp, str, pointer(arr), N, tcode,)
end

# This is the function that should be called by the end user
function ducc_u2nu(coord::StridedArray{Float64,2}, grid::StridedArray{T,N};
    forward = true,
    verbose = false,
    epsilon = 1e-5,
    nthreads = 1,
    periodicity = 2π) where {T,N}

    res = Vector{T}(undef,size(coord)[2])
# MR which variables do I need to preserve?
    GC.@preserve coord grid res ccall((:nufft_u2nu_julia_double_new, "./ducc_julia.so"), Cvoid, (ArrayDescriptor,ArrayDescriptor, Cint, Cdouble, Csize_t, ArrayDescriptor,Csize_t, Cdouble, Cdouble, Cdouble, Cint), ArrayDescriptor(grid), ArrayDescriptor(coord), 0, epsilon, nthreads, ArrayDescriptor(res), verbose, 1.1, 2.6, periodicity, 1)
    return res
end

# demo call
npoints=1000000
shp=(1000,1000)
coord = rand(Float64, length(shp),npoints) .- 0.5
grid = ones(Complex{Float32},shp)
res = ducc_u2nu(coord, grid)
