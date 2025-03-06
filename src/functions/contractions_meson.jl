@doc raw"""
    meson_connected_contractions(τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, Φ_kliₚ_t::AbstractArray, Φ_kliₚ_t₀::AbstractArray, Γ_arr::AbstractArray, iₚ::Integer, full_corr_matrix::Bool; neg_sign::Bool=true)

Contract the perambulators `τ₁_αkβl_t` and `τ₂_αkβl_t` and the mode doublets `Φ_kliₚ_t` and
`Φ_kliₚ_t₀` to get connected meson correlators where `τ₁_αkβlt` is used to propagate in
forward and `τ₂_αkβlt` in backward direction. These arrays are assumed to only contain data
for a single sink time `t` and source time `t₀`. The matrices in `Γ_arr` are the matrices in
the interpolating operators. If `full_corr_matrix == true` compute the full
correlator matrix \
`C_tnn̄ = <(Ψbar₂ Γ_n Ψ₁)(t, p) (Ψ₁ Γbar_n̄ Ψbar₂)(t₀, p)>` \
of them. Otherwise, only compute the diagonal matrix entries.

The momentum index `iₚ` is used for the momentum projection.

If `neg_sign=true` (default) multiply the correlator matrix with `-1` (convention).
"""
function meson_connected_contractions(
    τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray,
    Φ_kliₚ_t::AbstractArray, Φ_kliₚ_t₀::AbstractArray,
    Γ_arr::AbstractArray{<:AbstractMatrix}, iₚ::Integer, full_corr_matrix::Bool;
    neg_sign::Bool=true
)
    # Number of gamma matrices
    Nᵧ = length(Γ_arr)

    # Compute Γbar matrices and multiply with γ₅
    γ₅Γ_αβn = stack([γ[5]] .* Γ_arr)
    Γbarγ₅_αβn = stack([γ[4]] .* adjoint.(Γ_arr) .* [γ[4]*γ[5]])

    # Initialize ITensors
    #####################

    # Indices
    N_modes = size(τ₁_αkβlt, 2)
    α = IT.Index(4, "α")
    β = IT.Index(4, "β")
    k = IT.Index(N_modes, "k")
    l = IT.Index(N_modes, "l")
    n = IT.Index(Nᵧ, "n")
    n̄ = IT.Index(Nᵧ, "nbar")

    # Mode doublets at sink
    Φ_tiₚ = IT.itensor(@view(Φ_kliₚ_t[:, :, iₚ]), k, k')

    # Mode doublet at source   
    Φ_t₀iₚ = IT.itensor(@view(Φ_kliₚ_t₀[:, :, iₚ]), l, l')

    # Perambulators
    τ₁_t = IT.itensor(τ₁_αkβlt, α', k', β', l')
    τ₂_t = IT.itensor(τ₂_αkβlt, α, k, β, l)

    # Tensor contractions
    #####################

    # Precontractions
    τ₁Φτ₂Φ = (τ₁_t * conj(Φ_t₀iₚ)) * (conj(τ₂_t) * Φ_tiₚ)

    if full_corr_matrix
        γ₅Γ_n = IT.itensor(γ₅Γ_αβn, α, α', n)
        Γbarγ₅_n̄ = IT.itensor(Γbarγ₅_αβn, β', β, n̄)

        C_nn̄ = τ₁Φτ₂Φ * Γbarγ₅_n̄ * γ₅Γ_n

        # Sign flip flip
        if neg_sign
            C_nn̄ = -C_nn̄
        end
        return IT.array(C_nn̄, n, n̄)
    else
        # Initialize array for diagonal elements
        C_n = Array{ComplexF64}(undef, Nᵧ)

        # Loop over all gamma matrices
        for n in 1:Nᵧ
            γ₅Γ_n = IT.itensor(@view(γ₅Γ_αβn[:, :, n]), α, α')
            Γbarγ₅_n = IT.itensor(@view(Γbarγ₅_αβn[:, :, n]), β', β)

            C_n[n] = IT.scalar(τ₁Φτ₂Φ * γ₅Γ_n * Γbarγ₅_n)
        end

        # Sign flip flip
        if neg_sign
            C_n .= .-C_n
        end
        return C_n
    end
end

@doc raw"""
    generate_meson_connected_contract_func(Γ::AbstractMatrix, Γbar::AbstractMatrix) -> contract::Function

Generate a meson contraction functions of the form \
`C(x', x) = Γ_αα' * D₁⁻¹(x', x)_α'aβ'b * Γbar_β'β * D₂⁻¹(x', x)_βbαa` \
which takes the propagators `D₁⁻¹` and `D₂⁻¹` (with size (4, 3, 4, 3)) at fixed `x'`, `x`
and returns `C(x', x)`.
"""
function generate_meson_connected_contract_func(Γ::AbstractMatrix, Γbar::AbstractMatrix)
    Symbolics.@variables D₁⁻¹[1:4, 1:3, 1:4, 1:3], D₂⁻¹[1:4, 1:3, 1:4, 1:3]

    a = IT.Index(3, "a")
    b = IT.Index(3, "b")
    α = IT.Index(4, "α")
    β = IT.Index(4, "β")
    a = IT.Index(3, "a")
    b = IT.Index(3, "b")
    
    Γ_ = IT.itensor(Γ, α, α')
    Γbar_ = IT.itensor(Γbar, β', β)
    D₁⁻¹_ = IT.itensor(collect(D₁⁻¹), α', a, β', b)
    D₂⁻¹_ = IT.itensor(collect(D₂⁻¹), β, b, α, a)

    C = IT.scalar((Γ_ * D₁⁻¹_) * (Γbar_ * D₂⁻¹_))

    contract_real = eval(Symbolics.build_function(real(C), D₁⁻¹, D₂⁻¹))
    contract_imag = eval(Symbolics.build_function(imag(C), D₁⁻¹, D₂⁻¹))
    contract = (D₁⁻¹, D₂⁻¹) -> contract_real(D₁⁻¹, D₂⁻¹) + im*contract_imag(D₁⁻¹, D₂⁻¹)

    return contract
end

@doc raw"""
    meson_connected_sparse_contractions(τ₁_αkβl_t::AbstractArray, τ₂_αkβl_t::AbstractArray, sparse_modes_arrays_tt₀::NTuple{4, AbstractArray}, p_arr::AbstractVector{<:AbstractVector}, contract_arr::AbstractVecOrMat{<:Function}, full_corr_matrix::Bool; neg_sign::Bool=true) -> corr_matrix

Contract the perambulators `τ₁_αkβl_t` and `τ₂_αkβl_t` and the sparse Laplace modes in
`sparse_modes_arrays_tt₀` to get connected meson correlators where `τ₁_αkβlt` is used to
propagate in forward and `τ₂_αkβlt` in backward direction.
These arrays are assumed to only contain data for a single sink time `t` and source time
`t₀`. The resulting correlator matrix is of the form \
`C_tnn̄iₚ = <(Ψbar₂ Γ_n Ψ₁)(t, p) (Ψ₁ Γbar_n̄ Ψbar₂)(t₀, p)>` \
The array `p_arr` contains the integer momenta the correlator is projected to. It
corresponds to the last index in `C_tnn̄iₚ`. The array `contract_arr` contains the contraction
functions for the fermion propagators for a fixed source/sink position and time.
Generate these functions with the function `generate_meson_connected_contract_func`. 
If `full_corr_matrix == true` compute the full correlator matrix `C_tnn̄iₚ`. In that case
`contract_arr` must be a square matrix where every entry corresponds to on paire
`(Γ_n, Γbar_n̄)`. Otherwise, only compute the diagonal matrix entries. Then, `contract_arr`
must be a vector.

If `neg_sign=true` (default) multiply the correlator matrix with `-1` (convention).
"""
function meson_connected_sparse_contractions(
    τ₁_αkβl_t::AbstractArray, τ₂_αkβl_t::AbstractArray,
    sparse_modes_arrays_tt₀::NTuple{4, AbstractArray},
    p_arr::AbstractVector{<:AbstractVector},
    contract_arr::AbstractVecOrMat{<:Function}, full_corr_matrix::Bool; neg_sign::Bool=true
)
    # Unpack sparse modes arrays
    x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀ = sparse_modes_arrays_tt₀

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓ_t, 2)

    # Number of gamma matrices
    Nᵧ = size(contract_arr, 1)
    if full_corr_matrix && Nᵧ != size(contract_arr, 2)
        throw(ArgumentError("contract_arr is not a square matrix."))
    end

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Indices
    N_modes = size(τ₁_αkβl_t, 2)
    a = IT.Index(3, "a")
    b = IT.Index(3, "b")
    α = IT.Index(4, "α")
    β = IT.Index(4, "β")
    iₓ = IT.Index(N_points, "iₓ")
    k = IT.Index(N_modes, "k")
    l = IT.Index(N_modes, "l")
    n = IT.Index(Nᵧ, "n")
    n̄ = IT.Index(Nᵧ, "nbar")

    # Perambulator 1 in forward direction
    τ₁_t = IT.itensor(τ₁_αkβl_t, α, k, β, l)

    # Perambulator 2 in backward direction (indices: α, k, β, l)
    τ₂_t = IT.itensor(τ₂_αkβl_t, β', l, α', k)
    τ₂_bw_t = IT.itensor(γ[5], β, β') * conj(τ₂_t) * IT.itensor(γ[5], α', α)

    # Laplace modes for all iₓ'
    v_sink_t = IT.itensor(v_sink_ciₓk_t, a, iₓ', k)

    # Precontraction for smeared propagator 1
    vτ₁ = v_sink_t * τ₁_t

    # Precontraction for smeared propagator 2
    IT.replaceinds!(v_sink_t, (a, k) => (b, l))
    τv₂ = τ₂_bw_t * conj(v_sink_t)

    # Allocate correlator matrix (momentum is fastest changing index)
    if full_corr_matrix
        C_iₚnn̄ = zeros(ComplexF64, length(p_arr), Nᵧ, Nᵧ)
    else
        C_iₚn = zeros(ComplexF64, length(p_arr), Nᵧ)
    end

    # Preallocate smeared propagators
    D₁⁻¹_αaβbiₓ′_iₓ = Array{ComplexF64}(undef, 4, 3, 4, 3, N_points)
    D₁⁻¹_iₓ = IT.itensor(D₁⁻¹_αaβbiₓ′_iₓ, α, a, β, b, iₓ')
    D₂⁻¹_αaβbiₓ′_iₓ = Array{ComplexF64}(undef, 4, 3, 4, 3, N_points)
    D₂⁻¹_iₓ = IT.itensor(D₂⁻¹_αaβbiₓ′_iₓ, α, a, β, b, iₓ')

    # Loop over source position iₓ
    for iₓ in 1:N_points
        # Laplace modes at position iₓ
        v_src_iₓt₀ = IT.itensor(@view(v_src_ciₓk_t₀[:, iₓ, :]), b, l)

        # Smeared propagator 1 (forward direction)
        IT.mul!(D₁⁻¹_iₓ, vτ₁, conj(v_src_iₓt₀))

        # Smeared propagator 2 (backward direction)
        IT.replaceinds!(v_src_iₓt₀, (b, l) => (a, k))
        IT.mul!(D₂⁻¹_iₓ, v_src_iₓt₀, τv₂)

        # Contractions
        for iₓ′ in 1:N_points
            # Exponential for momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓ_t[:, iₓ′] - x_src_μiₓ_t₀[:, iₓ])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)

            D₁⁻¹_αaβb_iₓ′iₓ = @view(D₁⁻¹_αaβbiₓ′_iₓ[:, :, :, :, iₓ′])
            D₂⁻¹_αaβb_iₓiₓ′ = @view(D₂⁻¹_αaβbiₓ′_iₓ[:, :, :, :, iₓ′])
            
            if full_corr_matrix
                # Compute all correlator matrix entries
                for n in 1:Nᵧ, n̄ in 1:Nᵧ
                    C_iₚnn̄[:, n, n̄] += exp_mipΔx_arr * contract_arr[n, n̄](
                        D₁⁻¹_αaβb_iₓ′iₓ, D₂⁻¹_αaβb_iₓiₓ′
                    )
                end
            else
                # Compute diagonal correlator matrix entries
                for n in 1:Nᵧ
                    C_iₚn[:, n] += exp_mipΔx_arr * contract_arr[n](
                        D₁⁻¹_αaβb_iₓ′iₓ, D₂⁻¹_αaβb_iₓiₓ′
                    )
                end
            end
        end
    end

    if full_corr_matrix
        # Normalization
        C_iₚnn̄ .*= (prod(parms.Nₖ)/N_points)^2

        # Sign flip
        if neg_sign
            C_iₚnn̄ .*= -1
        end

        # Make momentum index the last
        return permutedims(C_iₚnn̄, (2, 3, 1))
    else
        # Normalization
        C_iₚn .*= (prod(parms.Nₖ)/N_points)^2

        # Sign flip
        if neg_sign
            C_iₚn .*= -1
        end

        # Make momentum index the last
        return permutedims(C_iₚn, (2, 1))
    end
end
