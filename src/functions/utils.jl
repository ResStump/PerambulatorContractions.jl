@doc raw"""
    parse_gamma_string(s)

Parse a string `s` and return the corresponding momomial of gamma matrices. The string can
contain the following symbols:
- "I" for the identity matrix
- "G1", "G2", "G3", "G4", "G5" for the `γ₁`, `γ₂`, `γ₃`, `γ₄`, `γ₅` matrices
- "sigmaμν" for the `σ_{μν} = i/2(γ_μ*γ_ν - γ_ν*γ_μ)` matrix
- "C" for the charge conjugation matrix
- "Pp", "Pm" for positive/negative parity projectors `0.5*(I + γ₄)`, `0.5*(I - γ₄)`
multiplied with together with "*". 

The prefactors "+", "-" and "i" (and combinations) are allowed.
"""
function parse_gamma_string(s)
    # Dictionary mapping strings to matrices
    gamma_dict = Dict(
        "I" => I,
        "C" => C,
        "Pp" => Pp,
        "Pm" => Pm,
        "G1" => γ[1],
        "G2" => γ[2],
        "G3" => γ[3],
        "G4" => γ[4],
        "G5" => γ[5]
    )

    # Add σ_{\mu\nu} terms
    for μ in 1:4, ν in (μ+1):4
        gamma_dict["sigma$(μ)$(ν)"] = σ_μν(μ, ν)
    end

    # Handle sign and imaginary unit
    prefactor = 1
    
    s = strip(s)
    if startswith(s, "-")
        prefactor *= -1
        s = strip(s[2:end])
    elseif startswith(s, "+")
        s = strip(s[2:end])  # Just remove the "+"
    end
    
    if startswith(s, "i")
        prefactor *= im
        s = strip(s[2:end])
    end

    # Split into terms
    s_arr = strip.(split(s, "*"))

    # Look up the corresponding matrices in the dictionary and multiply them
    Γ = prefactor*I
    for s_ in s_arr
        if haskey(gamma_dict, s_)
            Γ *= gamma_dict[s_]
        else
            error("Unknown symbol: $s_")
        end
    end

    return Γ
end

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
    N_points = size(x_sink_μiₓt, 2)

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
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

            # Indices
            N_modes = size(Φ_sink_kltiₚ, 1)
            a = IT.Index(3, "a")
            iₓ = IT.Index(N_points, "iₓ")
            k = IT.Index(N_modes, "k")
            l = IT.Index(N_modes, "l")

            # Allocate ITensors
            Φ_sink_tiₚ = IT.itensor(Φ_sink_kl_tiₚ, k, l)
            Φ_src_tiₚ = IT.itensor(Φ_src_kl_tiₚ, k, l)

            v_sink = IT.itensor(v_sink_ciₓk_t, a, iₓ, k)
            exp_v_sink = IT.itensor(exp_mipx_sink_iₓ .* v_sink_ciₓk_t, a, iₓ, l)
            v_src = IT.itensor(v_src_ciₓk_t, a, iₓ, k)
            exp_v_src = IT.itensor(exp_mipx_src_iₓ .* v_src_ciₓk_t, a, iₓ, l)

            # Tensor contraction
            IT.mul!(Φ_sink_tiₚ, conj(v_sink), exp_v_sink)
            IT.mul!(Φ_src_tiₚ, conj(v_src), exp_v_src)
        end
    end

    # Normalization
    Φ_sink_kltiₚ .*= prod(parms.Nₖ)/N_points
    Φ_src_kltiₚ .*= prod(parms.Nₖ)/N_points

    return
end

@doc raw"""
    scalar_triple_product(a::AbstractVector, b::AbstractVector, c::AbstractVector)

Compute the scalar triple product of three 3D vectors `a`, `b`, and `c`, given by 
`a ⋅ (b × c)`.
"""
function scalar_triple_product(a::AbstractVector, b::AbstractVector, c::AbstractVector)
    return @. a[1] * (b[2]*c[3] - b[3]*c[2]) +
              a[2] * (b[3]*c[1] - b[1]*c[3]) +
              a[3] * (b[1]*c[2] - b[2]*c[1])
end

@doc raw"""
    sparse_mode_triplets!(Φ_sink_Ktiₚ, Φ_src_Ktiₚ, sparse_modes_arrays, iₚ_arr, p_arr)

Compute the mode triplets from the sparse Laplace modes stored in `sparse_modes_arrays`.
The result is written to `Φ_sink_Ktiₚ` and `Φ_src_Ktiₚ` for the sink and the source 
respectively. Only the entries with the momentum indice in `iₚ_arr` for the corresponding
momentas in `p_arr` are computed. The remaining parts of `Φ_sink_Ktiₚ` and `Φ_src_Ktiₚ`
remain unaltered.
"""
function sparse_mode_triplets!(Φ_sink_Ktiₚ, Φ_src_Ktiₚ, sparse_modes_arrays, iₚ_arr, p_arr)
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    if size(iₚ_arr) != size(p_arr)
        throw(ArgumentError("the vectors `iₚ_arr` and `p_arr` don't have the same shape."))
    end
    
    # Number of points on spares lattice
    N_points = size(x_sink_μiₓt, 2)

    # Precomputation for momentum projection
    # (reshape to make p*x a vector matrix multiplication)
    mix_sink_arr = -2π*im * (x_sink_μiₓt./reshape(parms.Nₖ, (:, 1, 1)))
    mix_sink_arr = reshape(mix_sink_arr, (3, :))

    mix_src_arr = -2π*im * (x_src_μiₓt./reshape(parms.Nₖ, (:, 1, 1)))
    mix_src_arr = reshape(mix_src_arr, (3, :))

    # Permute vectors for memory efficiency
    v_sink_iₓtck = permutedims(v_sink_ciₓkt, [2, 4, 1, 3])
    v_src_iₓtck = permutedims(v_src_ciₓkt, [2, 4, 1, 3])

    # Compute sparse mode triplets at sink
    K = 1
    for k in 1:parms.N_modes
        v_sink_k_arr = eachslice(@view(v_sink_iₓtck[:, :, :, k]), dims=3)
        for l in k+1:parms.N_modes
            v_sink_l_arr = eachslice(@view(v_sink_iₓtck[:, :, :, l]), dims=3)
            for h in l+1:parms.N_modes
                v_sink_h_arr = eachslice(@view(v_sink_iₓtck[:, :, :, h]), dims=3)

                # Contract eigenmodes with epsilon tensor
                vvv_sink_iₓt =
                    scalar_triple_product(v_sink_k_arr, v_sink_l_arr, v_sink_h_arr)

                for (iₚ, p) in zip(iₚ_arr, p_arr)
                    # Compute exp(-ipx)
                    mipx_sink_iₓt = reshape(p' * mix_sink_arr, N_points, parms.Nₜ)
                    exp_mipx_sink_iₓt = exp.(mipx_sink_iₓt)

                    for iₜ in 1:parms.Nₜ
                        Φ_sink_Ktiₚ[K, iₜ, iₚ] =
                            transpose(@view(exp_mipx_sink_iₓt[:, iₜ])) * 
                            @view(vvv_sink_iₓt[:, iₜ])
                    end
                end

                K += 1
            end
        end
    end

    # Compute sparse mode triplets at src
    K = 1
    for k in 1:parms.N_modes
        v_src_k_arr = eachslice(@view(v_src_iₓtck[:, :, :, k]), dims=3)
        for l in k+1:parms.N_modes
            v_src_l_arr = eachslice(@view(v_src_iₓtck[:, :, :, l]), dims=3)
            for h in l+1:parms.N_modes
                v_src_h_arr = eachslice(@view(v_src_iₓtck[:, :, :, h]), dims=3)

                # Contract eigenmodes with epsilon tensor
                vvv_src_iₓt = scalar_triple_product(v_src_k_arr, v_src_l_arr, v_src_h_arr)

                for (iₚ, p) in zip(iₚ_arr, p_arr)
                    # Compute exp(-ipx)
                    mipx_src_iₓt = reshape(p' * mix_src_arr, N_points, parms.Nₜ)
                    exp_mipx_src_iₓt = exp.(mipx_src_iₓt)

                    for iₜ in 1:parms.Nₜ
                        Φ_src_Ktiₚ[K, iₜ, iₚ] = transpose(@view(exp_mipx_src_iₓt[:, iₜ])) * 
                            @view(vvv_src_iₓt[:, iₜ])
                    end
                end

                K += 1
            end
        end
    end

    # Normalization
    Φ_sink_Ktiₚ .*= prod(parms.Nₖ)/N_points
    Φ_src_Ktiₚ .*= prod(parms.Nₖ)/N_points

    return
end

@doc raw"""
    antisym_to_dense(Φ::AbstractVector) -> T::AbstractArray

Convert an antisymmetric rank 3 tensor `Φ` which is in the mode triplets format to a dense
rank 3 tensor `T`.
"""
function antisym_to_dense(Φ::AbstractVector)
    # Check size
    if length(parms.K_arr) != length(Φ)
        throw(ArgumentError("size of Φ not correct"))
    end

    # Allocate dense tensor
    T = zeros(ComplexF64, parms.N_modes, parms.N_modes, parms.N_modes)

    # Permutations in S₃ and their signs
    perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    signs = [1, -1, -1, 1, 1, -1]

    # Loop over the elements of Φ
    for K in eachindex(Φ)
        # Loop over the permutations
        for (sgn, perm) in zip(signs, perms)
            # Get the indices
            k, l, h = parms.K_arr[K][perm]

            # Store entry
            T[k, l, h] = sgn * Φ[K]
        end
    end

    return T
end

@doc raw"""
    antisym_contraction(Φ::AbstractVector, T::AbstractArray, idx::Int) -> AbstractArray

Compute the contraction of a tensor `T` along the `idx`-th dimension with a antisymmetric
rank 3 tensor `Φ` which is in the mode triplets format. The retuned tensor has the free
indices of `T` (in the same order) and the two antisymmetric indices as the last indices.
"""
function antisym_contraction(Φ::AbstractVector, T::AbstractArray, idx::Int)
    # Check if the index is valid
    if idx < 1 || idx > length(T)
        throw(ArgumentError("index out of bounds"))
    end

    # Check sizes
    if parms.N_modes != size(T, idx)
        throw(ArgumentError("size of Φ not correct"))
    end
    if length(parms.K_arr) != length(Φ)
        throw(ArgumentError("size of T not correct"))
    end

    # Make idx slowest index
    T = permutedims(T, ((1:idx-1)..., (idx+1:ndims(T)...), idx))
    res_size = (size(T)..., parms.N_modes)

    # Reshape to rank 2 tensor to use axpy!
    T_ = reshape(T, (:, parms.N_modes))

    # Allocate result
    T_res_ = zeros(ComplexF64, size(T_)..., parms.N_modes)
    
    # Permutations in S₃ and their signs
    perms = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    signs = [1, -1, -1, 1, 1, -1]

    # Loop over the elements of Φ
    for K in eachindex(Φ)
        # Loop over the permutations
        for (sgn, perm) in zip(signs, perms)
            # Get the indices
            k, l, h = parms.K_arr[K][perm]

            # Multiply tensors
            LA.axpy!(sgn * Φ[K], @view(T_[:, k]), @view(T_res_[:, l, h]))
        end
    end

    # Reshape to original shape
    T_res = reshape(T_res_, res_size)

    return T_res
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
