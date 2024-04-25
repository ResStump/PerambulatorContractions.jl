@doc raw"""
    increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new, n_cnfg)

Increase the separation of the sparse spaces in `sparse_modes_arrays` and store the result
in `sparse_modes_arrays_new`.

The new separation is `N_sep_new` which has to be larger than the current separation.
"""
function increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new)
    if mod(N_sep_new, 2) != 0
        throw(ArgumentError("'N_sep_new' has to be a multiple of 2."))
    end
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Determine number of points and seperation in current sparse modes
    N_points = size(x_sink_μiₓt)[2]
    N_sep = (prod(parms.Nₖ)/N_points)^(1/3)
    N_sep = round(Int, N_sep)

    # Create array with indices of new sparse modes
    iₓ_arr = collect(1:N_points)
    Nₖ_sparse = round(Int, N_points^(1/3))
    iₓ_arr = reshape(iₓ_arr, (Nₖ_sparse, Nₖ_sparse, Nₖ_sparse))
    division = N_sep_new÷N_sep
    if division == 0
        throw(ArgumentError("N_sep_new has to be bigger than the N_sep in "*
                            "sparse_modes_arrays."))
    end
    
    iₓ_new_arr = vec(iₓ_arr[1:division:end, 1:division:end, 1:division:end])

    x_sink_new_μiₓt, x_src_new_μiₓt, v_sink_new_ciₓkt, v_src_new_ciₓkt =
        sparse_modes_arrays_new

    # Fill new sparse spaces/modes at sink
    x_sink_new_μiₓt[:] = x_sink_μiₓt[:, iₓ_new_arr, :]
    v_sink_new_ciₓkt[:] = v_sink_ciₓkt[:, iₓ_new_arr, :, :]

    # Fill new sparse spaces/modes at src
    x_src_new_μiₓt[:] = x_src_μiₓt[:, iₓ_new_arr, :]
    v_src_new_ciₓkt[:] = v_src_ciₓkt[:, iₓ_new_arr, :, :]

    return
end

@doc raw"""
    increase_separation(sparse_modes_arrays, N_sep_new, n_cnfg)
        -> sparse_modes_arrays_new

Increase the separation of the sparse spaces in `sparse_modes_arrays`.

The new separation is `N_sep_new` which has to be larger than the current separation.
"""
function increase_separation(sparse_modes_arrays, N_sep_new)
    # Compute number of points in new sparse modes
    N_points=prod(parms.Nₖ)÷N_sep_new^3

    # Allocate arrays and compute new sparse modes
    sparse_modes_arrays_new = allocate_sparse_modes(N_points=N_points)
    increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays, N_sep_new)

    return sparse_modes_arrays_new
end
