import HDF5

include("allocate_arrays.jl")

@doc raw"""
    increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new, n_cnfg)

Increase the separation of the sparse spaces in `sparse_modes_arrays` and store the result
in `sparse_modes_arrays_new`.

The new separation is `N_sep_new` which has to be larger than the current separation.

`n_sep` is the number of the configuration the sparse mode has been computed from. It's used
to generate a different seed for each configuration.
"""
function increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new,
                              n_cnfg)
    if mod(N_sep_new, 2) != 0
        throw(ArgumentError("'N_sep_new' has to be a multiple of 2."))
    end
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Determine seperation in current sparse modes
    N_sep = (prod(parms.Nₖ)/size(x_sink_μiₓ)[2])^(1/3)
    N_sep = round(Int, N_sep)

    # Create array with indices of new sparse modes
    iₓ_arr = collect(1:size(x_sink_μiₓ)[2])
    Nₖ_sparse = round(Int, length(iₓ_arr)^(1/3))
    iₓ_arr = reshape(iₓ_arr, (Nₖ_sparse, Nₖ_sparse, Nₖ_sparse))
    iₓ_arr = permutedims(iₓ_arr, (3, 2, 1)) # Make row-major
    division = N_sep_new÷N_sep
    if division == 0
        throw(ArgumentError("'N_sep_new' has to be bigger than the N_sep in "*
                            "sparse_modes_arrays."))
    end

    seed = parms_toml["Increased Separation"]["seed"]
    seed_cnfg = seed ⊻ n_cnfg
    rng = Random.MersenneTwister(seed_cnfg)
    
    iₓ_sink_new_arr = vec(iₓ_arr[1:division:end, 1:division:end, 1:division:end])
    iₓ_src_new_arr = Array{Int}(undef, length(iₓ_sink_new_arr), parms.Nₜ)
    for iₜ in 1:parms.Nₜ
        offset = rand(rng, 1:division, 3)
        iₓ_src_new_arr[:, iₜ] = vec(iₓ_arr[offset[1]:division:end,
                                           offset[2]:division:end,
                                           offset[3]:division:end])
    end

    x_sink_new_μiₓ, x_src_new_μiₓt, v_sink_new_ciₓkt, v_src_new_ciₓkt =
        sparse_modes_arrays_new

    # Fill new sparse spaces/modes at sink
    x_sink_new_μiₓ[:] = x_sink_μiₓ[:, iₓ_sink_new_arr]
    v_sink_new_ciₓkt[:] = v_sink_ciₓkt[:, iₓ_sink_new_arr, :, :]

    # Fill new sparse spaces/modes at src
    for iₜ in 1:parms.Nₜ
        x_src_new_μiₓt[:, :, iₜ] = x_src_μiₓt[:, iₓ_src_new_arr[:, iₜ], iₜ]
        v_src_new_ciₓkt[:, :, :, iₜ] = v_src_ciₓkt[:, iₓ_src_new_arr[:, iₜ], :, iₜ]
    end

    return
end

@doc raw"""
    increase_separation!(sparse_modes_arrays, N_sep_new, n_cnfg)
        -> sparse_modes_arrays_new

Increase the separation of the sparse spaces in `sparse_modes_arrays`.

The new separation is `N_sep_new` which has to be larger than the current separation.

`n_sep` is the number of the configuration the sparse mode has been computed from. It's used
to generate a different seed for each configuration.
"""
function increase_separation(sparse_modes_arrays, N_sep_new, n_cnfg)
    # Compute number of points in new sparse modes
    N_points=prod(parms.Nₖ)÷N_sep_new^3

    # Allocate arrays and compute new sparse modes
    sparse_modes_arrays_new = allocate_sparse_modes(N_points=N_points)
    increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new,
                         n_cnfg)

    return sparse_modes_arrays_new
end
