using QuadGK
using LinearAlgebra
using Plots
using ITensors
include("tci_utilities.jl")
ITensors.disable_warn_order()

let

    # Define parameters and perform TCI
    N = 50
    d = 2
    num_variables = 2
    min_grid = 0
    max_grid = pi/2
    tolerance = 1e-14
    max_pivots = 200
    sweeps = 20
    num_starting_pivots = 100
    sites = siteinds(d, num_variables*N)

    # Redefine a function for the TCI
    function func_mps(dinary, min_grid, max_grid, d)

        """
        Convert a dinary vector to a grid value and then evaluate a given function with that input value
    
        Inputs:
        ditstring = dinary vector of integers between 1 and d e.g. [3, 2, 3, 1] for d = 3
        min_grid, max_grid = the grid domain over which values are mapped
    
        Return:
        x, y = grid value, function value at that grid value input
        """
    
        delta = (max_grid - min_grid)/(d^N - 1)
    
        x = min_grid + fixed_dinary_to_decimal(dinary[1:num_variables:end], d) * delta
        y = min_grid + fixed_dinary_to_decimal(dinary[2:num_variables:end], d) * delta
        # z = min_grid + fixed_dinary_to_decimal(dinary[3:num_variables:end], d) * delta

        # return x, 1/(1+x*y)
        return x, sin(x+y)
        # return x, sin(x)*sin(y)*sin(z)

    end
        
    # Initializing pivots and mps list
    row_pivots, col_pivots, func_cache = initialize_pivots_and_cache(func_mps, min_grid, max_grid, d, num_variables*N, num_starting_pivots)
    mps_list, row_pivots, col_pivots = get_mps_from_pivots(row_pivots, col_pivots, min_grid, max_grid, d, num_variables*N, func_cache, max_pivots, tolerance, func_mps)
    
    # Perform TCI
    mps_list, pivot_error_list, func_cache, sweep, bonddims = tci(num_variables*N, func_mps, min_grid, max_grid, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list)
    mps = mps_list_to_itensors_mps(num_variables*N, mps_list, sites)
    
    function integrate(mps, sites, d, N, num_variables)
        
        mps_int = MPS(sites)
        links = [Index(1, "Link,i=$n") for n in 1:length(mps)-1]
        vec = [1 for _ in 1:d]
        delta = (max_grid - min_grid)/(d^N - 1)        
        for i in 1:N*num_variables
            if i == 1
                mps_int[i] = ITensor(reshape(vec, d, 1), sites[i], links[i])
            elseif i == N*num_variables
                mps_int[i] = ITensor(reshape(vec, d, 1), sites[i], links[i-1])
            else
                mps_int[i] = ITensor(reshape(vec, d, 1, 1), sites[i], links[i-1], links[i])
            end
        end
        res = inner(mps_int, mps)*delta^num_variables

        return res
    
    end

    # Get integration result
    res = integrate(mps, sites, d, N, num_variables)
    println(abs(res-2))
    
    println("\nFinished.")
    
end
