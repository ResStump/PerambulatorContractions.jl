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
    x_sink_new_μiₓt .= x_sink_μiₓt[:, iₓ_new_arr, :]
    v_sink_new_ciₓkt .= v_sink_ciₓkt[:, iₓ_new_arr, :, :]

    # Fill new sparse spaces/modes at src
    x_src_new_μiₓt .= x_src_μiₓt[:, iₓ_new_arr, :]
    v_src_new_ciₓkt .= v_src_ciₓkt[:, iₓ_new_arr, :, :]

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

@doc raw"""
    sparse_mode_doublets!(Φ_sink_kltiₚ, Φ_src_kltiₚ, sparse_modes_arrays, iₚ_arr, p_arr)

Compute the mode doublets from the sparse Laplace modes stored in `sparse_modes_arrays`.
The result is written to `Φ_sink_kltiₚ` and `Φ_src_kltiₚ` for the sink and the source 
respectively. Only the entries with the momentum indice in `iₚ_arr` for the corresponding
momentas in `p_arr` are computed. The remaining parts of `Φ_sink_kltiₚ` and `Φ_src_kltiₚ`
remain unaltered.
"""
function sparse_mode_doublets!(Φ_sink_kltiₚ, Φ_src_kltiₚ, sparse_modes_arrays,
                               iₚ_arr, p_arr)
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    if size(iₚ_arr) != size(p_arr)
        throw(ArgumentError("the vectors `iₚ_arr` and `p_arr` don't have the same shape."))
    end
    
    # Number of points on spares lattice
    _, N_points, _ = size(x_sink_μiₓt)

    # Loop over all sink time indice
    Threads.@threads for iₜ in 1:parms.Nₜ
        for (iₚ, p) in zip(iₚ_arr, p_arr)
            # Mode doublet and Laplace modes at sink time t (index `iₜ`)
            Φ_sink_kl_tiₚ = @view Φ_sink_kltiₚ[:, :, iₜ, iₚ]
            x_sink_μiₓ_t = @view x_sink_μiₓt[:, :, iₜ]
            v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

            # Mode doublet and Laplace modes at src time t (index `iₜ`)
            Φ_src_kl_tiₚ = @view Φ_src_kltiₚ[:, :, iₜ, iₚ]
            x_src_μiₓ_t = @view x_src_μiₓt[:, :, iₜ]
            v_src_ciₓk_t = @view v_src_ciₓkt[:, :, :, iₜ]

            # Compute exp(-ipx) for sink and reshape it to match shape of Laplace modes
            exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ_t./parms.Nₖ)'*p)
            exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))

            # Compute exp(-ipx) for src and reshape it to match shape of Laplace modes
            exp_mipx_src_iₓ = exp.(-2π*im * (x_src_μiₓ_t./parms.Nₖ)'*p)
            exp_mipx_src_iₓ = reshape(exp_mipx_src_iₓ, (1, N_points, 1))

            # Tensor contraction
            TO.@tensoropt (a=>3, l=>32, k=>32, iₓ=>512) begin
                Φ_sink_kl_tiₚ[k, l] = 
                    conj(v_sink_ciₓk_t[a, iₓ, k]) * 
                    (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ, l]

                Φ_src_kl_tiₚ[k, l] = 
                    conj(v_src_ciₓk_t[a, iₓ, k]) * 
                    (exp_mipx_src_iₓ .* v_src_ciₓk_t)[a, iₓ, l]
            end
        end
    end

    return
end

@doc raw"""
    generate_momentum_pairs(Ptot_sq, p_sq_sum_max; ret_Ptot=false)

Compute an array of momentum indices pairs for which the two momenta (p₁, p₂) fulfil the 
following two properties:
- (p₁ + p₂)² = `Ptot_sq`
- p₁² + p₂² ≤ `p_sq_sum`
- If (p₂, p₁) is already in the array, then (p₁, p₂) is not added (relevant for p₁ ≠ p₂)
- p₁ = p₂ is allowed

The available momentum indices and the corresponding momenta are taken from `parms`.

If `ret_Ptot=true` the array of total momenta is also returned.
"""
function generate_momentum_pairs(Ptot_sq, p_sq_sum_max; ret_Ptot=false)
    # Compute all total momenta with P² = Ptot_sq
    Ptot_arr = []
    Pᵢ_max = round(Int, √Ptot_sq)
    for (Px, Py, Pz) in Iterators.product(-Pᵢ_max:Pᵢ_max, -Pᵢ_max:Pᵢ_max, -Pᵢ_max:Pᵢ_max)
        P = [Px, Py, Pz]
        if P'*P == Ptot_sq
            push!(Ptot_arr, P)
        end
    end

    # Compute all valid pairs of momenta and store their indices
    Iₚ_arr = []
    for Ptot in Ptot_arr
        # Loop over all combinations of two momentum indeces
        # (twice the same momentum is allowed) 
        for (iₚ₁, iₚ₂) in Comb.with_replacement_combinations(parms.iₚ_arr, 2)
            p₁, p₂ = parms.p_arr[[iₚ₁, iₚ₂]]
            # Only use momenta p₁, p₂ if p₁² + p₂² small enough and tot. momentum is correct
            if p₁'*p₁ + p₂'*p₂ <= p_sq_sum_max && p₁ + p₂ == Ptot
                push!(Iₚ_arr, [iₚ₁, iₚ₂])
            end
        end
    end

    if ret_Ptot
        return Iₚ_arr, Ptot_arr
    else
        return Iₚ_arr
    end
end

@doc raw"""
    generate_momentum_4tuples(Ptot_sq, p_sq_sum_max; ret_Ptot=false)

Compute an array of momentum indices 4-tuples for which the four momenta (p₁, p₂, p₃, p₄)
fulfil the following properties:
- (p₁ + p₂)² = (p₃ + p₄)² = `Ptot_sq`
- p₁² + p₂² ≤ `p_sq_sum` and p₃² + p₄² ≤ `p_sq_sum`
- If (p₂, p₁) is already in the array, then (p₁, p₂) is not added (relevant for p₁ ≠ p₂) \
  (the same for (p₃, p₄))
- p₁ = p₂ or p₃ = p₄ is allowed

The available momentum indices and the corresponding momenta are taken from `parms`.

If `ret_Ptot=true` the array of total momenta is also returned.
"""
function generate_momentum_4tuples(Ptot_sq, p_sq_sum_max; ret_Ptot=false)
    # Compute all total momenta with P² = Ptot_sq
    Ptot_arr = []
    Pᵢ_max = round(Int, √Ptot_sq)
    for (Px, Py, Pz) in Iterators.product(-Pᵢ_max:Pᵢ_max, -Pᵢ_max:Pᵢ_max, -Pᵢ_max:Pᵢ_max)
        P = [Px, Py, Pz]
        if P'*P == Ptot_sq
            push!(Ptot_arr, P)
        end
    end

    # Compute all valid 4-tuple of momenta and store their indices
    Iₚ_arr = []
    for Ptot in Ptot_arr
        # Loop over all combinations of two momentum indeces
        # (twice the same momentum is allowed) 
        Iₚ_pair_arr = []
        for (iₚ₁, iₚ₂) in Comb.with_replacement_combinations(parms.iₚ_arr, 2)
            p₁, p₂ = parms.p_arr[[iₚ₁, iₚ₂]]
            # Only use momenta p₁, p₂ if p₁² + p₂² small enough and tot. momentum is correct
            if p₁'*p₁ + p₂'*p₂ <= p_sq_sum_max && p₁ + p₂ == Ptot
                push!(Iₚ_pair_arr, [iₚ₁, iₚ₂])
            end
        end

        # Get all combinations of pairs
        for (Iₚ_pair_1, Iₚ_pair_2) in Iterators.product(Iₚ_pair_arr, Iₚ_pair_arr)
            push!(Iₚ_arr, [Iₚ_pair_1..., Iₚ_pair_2...])
        end
    end

    if ret_Ptot
        return Iₚ_arr, Ptot_arr
    else
        return Iₚ_arr
    end
end
