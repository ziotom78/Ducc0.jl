include("/home/martin/codes/Ducc0.jl/src/Ducc0.jl")
using .Ducc0.Nufft: u2nu!, nu2u!, make_plan, delete_plan!, u2nu_planned, nu2u_planned
using Test

@testset "Nufft" begin
    let ndim = 3, npoints = 10_000_000, shape = (100, 200, 300)
        grid = Array{Complex{Float64}}(undef, shape)
        coord = Array{Float64}(undef, (ndim, npoints))
        points = Array{Complex{Float64}}(undef, (npoints,))

        u2nu!(coord, grid, points, sigma_min=1.1, sigma_max=2.5, forward = true, verbose = true, epsilon = 1e-5, nthreads=Unsigned(4))
        nu2u!(coord, points, grid, sigma_min=1.1, sigma_max=2.5, forward = true, verbose = true, epsilon = 1e-5, nthreads=Unsigned(4))
    end
end

@testset "Nufft.plan" begin
    let ndim = 3, npoints = 10_000_000, shape = (100, 200, 300)
        grid = Array{Complex{Float64}}(undef, shape)
        coord = Array{Float64}(undef, (ndim, npoints))
        points = Array{Complex{Float64}}(undef, (npoints,))

        plan = make_plan(coord, shape, sigma_min=1.1, sigma_max=2.6, epsilon=1e-5, nthreads=Unsigned(4), periodicity=2π)
        u2nu_planned(plan, grid, forward=true, verbose=true)
        delete_plan!(plan)
        
        plan = make_plan(coord, shape, sigma_min=1.1, sigma_max=2.6, epsilon=1e-5, nthreads=Unsigned(4), periodicity=2π)
        nu2u_planned(plan, points, forward=true, verbose=true)
        delete_plan!(plan)
    end
end
