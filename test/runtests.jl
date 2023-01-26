include("/home/martin/codes/Ducc0.jl/src/Ducc0.jl")
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
    arr2 = Ducc0.Fft.r2r_genuine_hartley(Ducc0.Fft.r2r_genuine_hartley(arr, (1,2,3), fct=1.), (2,3,1), fct=1.0/(100*200*300))
    res = LinearAlgebra.norm(arr-arr2)/LinearAlgebra.norm(arr)
    @test res <= 1e-14
end
@testset "Sht" begin
    lmax = 100
    theta = Vector{Float64}(undef, 2*lmax+2)
    for i=1:2*lmax+2
        theta[i] = (i-1)*π/(2*lmax+1)
    end
    mval = Vector{Csize_t}(undef, lmax+1)
    mstart = Vector{Cptrdiff_t}(undef, lmax+1)
    ofs=0
    for i=1:lmax+1
        mval[i] = i-1
        mstart[i] = ofs
        ofs += lmax+1-i-1
    end
    println(mstart[lmax+1])
    alm=zeros(Complex{Float64}, mstart[lmax+1]+10,1)
    Ducc0.Sht.alm2leg(alm, 0, lmax, mval, mstart, 1, theta, 0)
end
