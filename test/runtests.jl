using Ducc0
using Test

@testset "Nufft" begin
    let ndim = 3, npoints = 10_000_000, shape = [100, 200, 300]
        grid = Array{Float64}(undef, prod(shape) * 2)
        coord = Array{Float64}(undef, ndim * npoints)
        points = Array{Float64}(undef, npoints * 2)

        nufft_u2nu!(ndim, npoints, shape, grid, coord, points, 1.1, 2.5, 0, forward = true, verbosity = true, epsilon = 1e-5, nthreads=4)
        nufft_nu2u!(ndim, npoints, shape, points, coord, grid, 1.1, 2.5, 0, forward = true, verbosity = true, epsilon = 1e-5, nthreads=4)
    end
end

@testset "Nufft.plan" begin
    let ndim = 3, npoints = 10_000_000, shape = [100, 200, 300]
        grid = Array{Float64}(undef, prod(shape) * 2)
        coord = Array{Float64}(undef, ndim * npoints)
        points = Array{Float64}(undef, npoints * 2)

        plan = make_nufft_plan(0, ndim, npoints, shape, coord, 1.1, 2.6, 0, epsilon=1e-5, nthreads=4, periodicity=2π)
        planned_u2nu(plan, grid, points, forward=true, verbosity=true)
        delete_nufft_plan(plan)
        
        plan = make_nufft_plan(1, ndim, npoints, shape, coord, 1.1, 2.6, 0, epsilon=1e-5, nthreads=4, periodicity=2π)
        planned_nu2u(plan, points, grid, forward=true, verbosity=true)
        delete_nufft_plan(plan)
    end
end
