#include("/home/martin/codes/Ducc0.jl/src/Ducc0.jl")
include("../src/Ducc0.jl")
using .Ducc0.Nufft: u2nu!, nu2u!, make_plan, delete_plan!, u2nu_planned, nu2u_planned
using Test

@testset "Nufft" begin
    let ndim = 3, npoints = 10_000_000, shape = (100, 200, 300)
        grid = Array{Complex{Float64}}(undef, shape)
        coord = Array{Float64}(undef, (ndim, npoints))
        points = Array{Complex{Float64}}(undef, (npoints,))

        u2nu!(coord, grid, points, sigma_min=1.1, sigma_max=2.5, forward = true, verbose = true, epsilon = 1e-5, nthreads=4)
        nu2u!(coord, points, grid, sigma_min=1.1, sigma_max=2.5, forward = true, verbose = true, epsilon = 1e-5, nthreads=4)
    end
end

@testset "Nufft.plan" begin
    let ndim = 3, npoints = 10_000_000, shape = (100, 200, 300)
        grid = Array{Complex{Float64}}(undef, shape)
        coord = Array{Float64}(undef, (ndim, npoints))
        points = Array{Complex{Float64}}(undef, (npoints,))

        plan = make_plan(coord, shape, sigma_min=1.1, sigma_max=2.6, epsilon=1e-5, nthreads=4, periodicity=2π)
        u2nu_planned(plan, grid, forward=true, verbose=true)
        delete_plan!(plan)

        plan = make_plan(coord, shape, sigma_min=1.1, sigma_max=2.6, epsilon=1e-5, nthreads=4, periodicity=2π)
        nu2u_planned(plan, points, forward=true, verbose=true)
        delete_plan!(plan)
    end
end

@testset "Fft" begin
    import Random
    import LinearAlgebra
    arr = Array{Complex{Float64}}(undef, (100,200,300))
    Random.rand!(arr)
    arr2 = Ducc0.Fft.c2c(Ducc0.Fft.c2c(arr, (1,3), forward=true, fct=1.), (3,1), forward=false, fct=1.0/(100*300))
    res = LinearAlgebra.norm(arr-arr2)/LinearAlgebra.norm(arr)
    @test res <= 1e-14
    arr = Array{Float64}(undef, (100,200,300))
    Random.rand!(arr)
    arr2 = Ducc0.Fft.r2r_genuine_fht(Ducc0.Fft.r2r_genuine_fht(arr, (1,2,3), fct=1.), (2,3,1), fct=1.0/(100*200*300))
    res = LinearAlgebra.norm(arr-arr2)/LinearAlgebra.norm(arr)
    @test res <= 1e-14
end

@testset "Sht" begin
    lmax = 100
    mmax = 100
    import Healpix
    hmap = Healpix.HealpixMap{Float64, Healpix.RingOrder}(64)
    rings = [r for r in 1:Healpix.numOfRings(hmap.resolution)]
    theta = Vector{Float64}(undef, length(rings)) #colatitude of every ring
    phi0 = Vector{Cdouble}(undef, length(rings))  #longitude of the first pixel of every ring
    nphi = Vector{Csize_t}(undef, length(rings))  #num of pixels in every ring
    rstart = Vector{Csize_t}(undef, length(rings)) #index of the first pixel in every ring
    rinfo = Healpix.RingInfo(0, 0, 0, 0, 0)               #initialize ring info object
    for ri in 1:length(rings)
        Healpix.getringinfo!(hmap.resolution, ri, rinfo)  #fill it
        theta[ri] = rinfo.colatitude_rad            #use it to get the necessary data
        phi0[ri] = Healpix.pix2ang(hmap, rinfo.firstPixIdx)[2]
        nphi[ri] = rinfo.numOfPixels
        rstart[ri] = rinfo.firstPixIdx
    end
    mval = Vector{Csize_t}(undef, lmax+1)
    mstart = Vector{Cptrdiff_t}(undef, lmax+1)
    ofss = 1
    for i in 1:lmax+1
        mval[i] = Csize_t(i-1)
        mstart[i] = Cptrdiff_t(ofss)
        ofss += lmax+1-i-1
    end
    println(mstart[lmax+1])
    # alm <-> leg
    alm_in = zeros(Complex{Float64}, maximum(mstart) + lmax,1)
    leg_in = Ducc0.Sht.alm2leg(alm_in, 0, lmax, mval, mstart, 1, theta, 0)
    alm_out = Ducc0.Sht.leg2alm(leg_in, 0, lmax, mval, mstart, 1, theta, 0)
    @test size(alm_in) == size(alm_out)

    # leg <-> map
    map_in = Ducc0.Sht.leg2map(leg_in, nphi, phi0, rstart, 1, 0)
    leg_out = Ducc0.Sht.map2leg(map_in, nphi, phi0, rstart, mmax, 1, 0)
    @test size(leg_in) == size(leg_out)
end
