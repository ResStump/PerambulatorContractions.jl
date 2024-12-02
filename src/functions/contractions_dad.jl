@doc raw"""
    dad_local_contractons!(C_tnmiₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ₁_arr::AbstractVector{<:AbstractMatrix}, Γ₂_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, p_arr::AbstractVector{<:AbstractVector})

Contract the charm perambulator `τ_charm_αkβlt` and the light perambulator `τ_light_αkβlt`
and the sparse Laplace modes in `sparse_modes_arrays` to get the local diquark-antidiquark
correlator. The matrices in `Γ₁_arr` and `Γ₂_arr` are the matrices in the interpolating
operators. The correlator is computed for all possible combinations of them. This gives a
vacuum expectation value of the form \
`ε\_{abc} ε\_{ade} ε\_{a'b'c'} ε\_{a'd'e'} \
<(c\_b^T CΓ₁ c\_c  ̄u\_d CΓ₂ d̄\_e^T)(x')
(c̄\_b' C ΓbarC₁ c̄\_c'^T  d\_d'^T C ΓbarC₂ u\_e')(x)>` \
(in position space) where `ΓbarCᵢ = Cγ₄ Γᵢ^† γ₄C`. This is computed for all combinations of
the gamma matrices in `Γ₁_arr` and `Γ₂_arr`. The result is stored in `C_tnmiₚ`
where the indices n, m, correspond to the indices of the Γ's in the expectation value
in the given order.

The source time `t₀` is used to circularly shift `C_tnmiₚ` such that it is at
the origin. The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in `C_tnmiₚ`).
"""
function dad_local_contractons!(
    C_tnmiₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, Γ₁_arr::AbstractVector{<:AbstractMatrix},
    Γ₂_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer,
    p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓt, 2)

    # Number of colors (could probably just use 3)
    N_c = size(v_sink_ciₓkt, 1)

    # Number of gamma matrices
    Nᵧ_1 = length(Γ₁_arr)
    Nᵧ_2 = length(Γ₂_arr)

    # Convert vector of γ-matrices to contiguous array and compute CΓ and CΓbarC matrices
    Γ₁_αβn = stack(Γ₁_arr)
    Γ₂_αβn = stack(Γ₂_arr)
    TO.@tensoropt CΓ₁_αβn[α, β, n] := C[α, α'] * Γ₁_αβn[α', β, n]
    TO.@tensoropt CΓ₂_αβn[α, β, n] := C[α, α'] * Γ₂_αβn[α', β, n]
    TO.@tensoropt CΓbarC₁_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₁_αβn)[β', α'', n] * γ[2][β', β]
    TO.@tensoropt CΓbarC₂_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₂_αβn)[β', α'', n] * γ[2][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Set correlator C_tnmiₚ to zero and permute such that time is slowest changing index
    C_tnmiₚ .= 0
    C_nmiₚt = permutedims(C_tnmiₚ, [2, 3, 4, 1])

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Time index for storing correlator entrie
        i_Δt = mod1(iₜ-t₀, parms.Nₜ)

        # Perambulators at sink time t
        τ_charm_αkβl_t = @view τ_charm_αkβlt[:, :, :, :, iₜ]
        τ_light_αkβl_t = @view τ_light_αkβlt[:, :, :, :, iₜ]

        # Conjugate perambulator and multiply γ₅ to use γ₅-hermiticity
        TO.@tensoropt (l, k) begin
            γ₅τ_conjγ₅_light_αkβl_t[α, k, β, l] :=
                γ[5][α, α'] * conj(τ_light_αkβl_t)[α', k, β', l] * γ[5][β', β]
        end

        # Loop over source position iₓ
        for iₓ in 1:N_points
            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Precontaction for smeared charm propagator
            TO.@tensoropt (k, l) begin
                τv_charm_kαβb[k, α, β, b] :=
                    τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
            end

            # Precontaction for smeared light propagator
            TO.@tensoropt (k, l) begin
                τv_light_kαβa[l, α, β, a] :=
                    γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * v_src_ck_iₓt₀[a, k]
            end

            # Loop over sink position iₓ′
            for iₓ′ in 1:N_points
                # Laplace modes at sink time t and position iₓ′
                v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

                # Tensor contractions
                #####################

                # Smeared charm propagator (forward direction)
                TO.@tensoropt (k, ) begin
                    D⁻¹_charm_αaβb_iₓ′iₓ[α, a, β, b] :=
                        v_sink_ck_iₓ′t[a, k] * τv_charm_kαβb[k, α, β, b]
                end
                #= TO.@tensoropt (k, l) begin
                    D⁻¹_charm_αaβb_iₓ′iₓ_[α, a, β, b] :=
                        v_sink_ck_iₓ′t[a, k] * 
                        τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
                end
                @assert D⁻¹_charm_αaβb_iₓ′iₓ ≈ D⁻¹_charm_αaβb_iₓ′iₓ_ =#

                # Smeared light propagator (backward direction)
                TO.@tensoropt (k, ) begin
                    D⁻¹_light_αaβb_iₓiₓ′[α, a, β, b] :=
                        conj(v_sink_ck_iₓ′t)[b, l] * τv_light_kαβa[l, α, β, a]
                end
                #= TO.@tensoropt (k, l) begin
                    D⁻¹_light_αaβb_iₓiₓ′_[α, a, β, b] :=
                        conj(v_sink_ck_iₓ′t)[b, l] *
                        γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * v_src_ck_iₓt₀[a, k]
                end
                @assert D⁻¹_light_αaβb_iₓiₓ′ ≈ D⁻¹_light_αaβb_iₓiₓ′_ =#

                # Light part
                C_light_dd′ee′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_2)
                for m in 1:Nᵧ_2
                    C_light_dd′ee′_m = @view C_light_dd′ee′n[:, :, :, :, m]
                    CΓ₂_αβ_m = @view CΓ₂_αβn[:, :, m]
                    CΓbarC₂_αβ_m = @view CΓbarC₂_αβn[:, :, m]
                    TO.@tensoropt begin
                        C_light_dd′ee′_m[d, d', e, e'] =
                            CΓ₂_αβ_m[δ, ϵ] * D⁻¹_light_αaβb_iₓiₓ′[δ', d', ϵ, e] *
                            CΓbarC₂_αβ_m[δ', ϵ'] * D⁻¹_light_αaβb_iₓiₓ′[ϵ', e', δ, d]
                    end
                end

                # Positive charm part
                C_pos_charm_bb′cc′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_1)
                for n in 1:Nᵧ_1
                    C_pos_charm_bb′cc′_n = @view C_pos_charm_bb′cc′n[:, :, :, :, n]
                    CΓ₁_αβ_n = @view CΓ₁_αβn[:, :, n]
                    CΓbarC₁_αβ_n = @view CΓbarC₁_αβn[:, :, n]
                    TO.@tensoropt begin
                        C_pos_charm_bb′cc′_n[b, b', c, c'] =
                            CΓ₁_αβ_n[β, γ] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, β', b'] *
                            CΓbarC₁_αβ_n[β', γ'] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, γ', c']
                    end
                end

                # Negative charm part
                C_neg_charm_bb′cc′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_1)
                for n in 1:Nᵧ_1
                    C_neg_charm_bb′cc′_n = @view C_neg_charm_bb′cc′n[:, :, :, :, n]
                    CΓ₁_αβ_n = @view CΓ₁_αβn[:, :, n]
                    CΓbarC₁_αβ_n = @view CΓbarC₁_αβn[:, :, n]
                    TO.@tensoropt begin
                        C_neg_charm_bb′cc′_n[b, b', c, c'] =
                            CΓ₁_αβ_n[β, γ] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, γ', c'] *
                            CΓbarC₁_αβ_n[β', γ'] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, β', b']
                    end
                end

                # Combine light and charm parts (sum over epsilon tensors)
                TO.@tensoropt begin
                    C_nm[n, m] :=
                        (
                            C_light_dd′ee′n[b, b', c, c', m] +
                            C_light_dd′ee′n[c, c', b, b', m] -
                            C_light_dd′ee′n[b, c', c, b', m] -
                            C_light_dd′ee′n[c, b', b, c', m]
                        ) *
                        (
                            C_pos_charm_bb′cc′n[b, b', c, c', n] -
                            C_neg_charm_bb′cc′n[b, b', c, c', n]
                        )
                end

                # Momentum projection
                m2πiΔx = -2π*im * 
                    (x_sink_μiₓt[:, iₓ′, iₜ] - x_src_μiₓt[:, iₓ, i_t₀])./parms.Nₖ
                exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
                for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                    # Use Δt=t-t₀ as time
                    C_nm_iₚΔt = @view C_nmiₚt[:, :, iₚ, i_Δt]
                    TO.@tensoropt begin
                        C_nm_iₚΔt[n, m] += exp_mipΔx * C_nm[n, m]
                    end
                end
            end
        end
    end

    # Permute correlator back
    permutedims!(C_tnmiₚ, C_nmiₚt, [4, 1, 2, 3])

    # Normalization
    C_tnmiₚ .*= (prod(parms.Nₖ)/N_points)^2

    return
end

@doc raw"""
    dad_local_contractons(τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray, sparse_modes_arrays_tt₀::NTuple{4, AbstractArray}, Γ₁_arr::AbstractVector{<:AbstractMatrix}, Γ₂_arr::AbstractVector{<:AbstractMatrix}, p_arr::AbstractVector{<:AbstractVector}) -> C_nmiₚ::AbstractArray

Contract the charm perambulator `τ_charm_αkβl_t` and the light perambulator `τ_light_αkβl_t`
and the sparse Laplace modes in `sparse_modes_arrays_tt₀` to get the local
diquark-antidiquark correlator and return it. These arrays are assumed to only contain data
for a single sink time `t` and source time `t₀`. The matrices in `Γ₁_arr` and `Γ₂_arr` are
the matrices in the interpolating operators. The correlator is computed for all possible
combinations of them. This gives a vacuum expectation value of the form \
`ε\_{abc} ε\_{ade} ε\_{a'b'c'} ε\_{a'd'e'} \
<(c\_b^T CΓ₁ c\_c  ̄u\_d CΓ₂ d̄\_e^T)(x')
(c̄\_b' C ΓbarC₁ c̄\_c'^T  d\_d'^T C ΓbarC₂ u\_e')(x)>` \
(in position space) where `ΓbarCᵢ = Cγ₄ Γᵢ^† γ₄C`. This is computed for all combinations of
the gamma matrices in `Γ₁_arr` and `Γ₂_arr`. The result is retuned as the array `C_nmiₚ`
where the indices n, m, correspond to the indices of the Γ's in the expectation value
in the given order.

The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in the returned array `C_nmiₚ`).
"""
function dad_local_contractons(
    τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray,
    sparse_modes_arrays_tt₀::NTuple{4, AbstractArray},
    Γ₁_arr::AbstractVector{<:AbstractMatrix}, Γ₂_arr::AbstractVector{<:AbstractMatrix},
    p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀ = sparse_modes_arrays_tt₀

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓ_t, 2)

    # Number of colors (could probably just use 3)
    N_c = size(v_sink_ciₓk_t, 1)

    # Number of gamma matrices
    Nᵧ_1 = length(Γ₁_arr)
    Nᵧ_2 = length(Γ₂_arr)

    # Convert vector of γ-matrices to contiguous array and compute CΓ and CΓbarC matrices
    Γ₁_αβn = stack(Γ₁_arr)
    Γ₂_αβn = stack(Γ₂_arr)
    TO.@tensoropt CΓ₁_αβn[α, β, n] := C[α, α'] * Γ₁_αβn[α', β, n]
    TO.@tensoropt CΓ₂_αβn[α, β, n] := C[α, α'] * Γ₂_αβn[α', β, n]
    TO.@tensoropt CΓbarC₁_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₁_αβn)[β', α'', n] * γ[2][β', β]
    TO.@tensoropt CΓbarC₂_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₂_αβn)[β', α'', n] * γ[2][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Allocate correlator
    C_nmiₚ = zeros(ComplexF64, Nᵧ_1, Nᵧ_2, length(p_arr))

    # Conjugate perambulator and multiply γ₅ to use γ₅-hermiticity
    TO.@tensoropt (l, k) begin
        γ₅τ_conjγ₅_light_αkβl_t[α, k, β, l] :=
            γ[5][α, α'] * conj(τ_light_αkβl_t)[α', k, β', l] * γ[5][β', β]
    end

    # Loop over source position iₓ
    for iₓ in 1:N_points
        # Laplace modes at position iₓ
        v_src_ck_iₓt₀ = @view v_src_ciₓk_t₀[:, iₓ, :]

        # Precontaction for smeared charm propagator
        TO.@tensoropt (k, l) begin
            τv_charm_kαβb[k, α, β, b] :=
                τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
        end

        # Precontaction for smeared light propagator
        TO.@tensoropt (k, l) begin
            τv_light_kαβa[l, α, β, a] :=
                γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * v_src_ck_iₓt₀[a, k]
        end

        # Loop over sink position iₓ′
        for iₓ′ in 1:N_points
            # Laplace modes at position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓk_t[:, iₓ′, :]

            # Tensor contractions
            #####################

            # Smeared charm propagator (forward direction)
            TO.@tensoropt (k, ) begin
                D⁻¹_charm_αaβb_iₓ′iₓ[α, a, β, b] :=
                    v_sink_ck_iₓ′t[a, k] * τv_charm_kαβb[k, α, β, b]
            end
            #= TO.@tensoropt (k, l) begin
                D⁻¹_charm_αaβb_iₓ′iₓ_[α, a, β, b] :=
                    v_sink_ck_iₓ′t[a, k] * 
                    τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
            end
            @assert D⁻¹_charm_αaβb_iₓ′iₓ ≈ D⁻¹_charm_αaβb_iₓ′iₓ_ =#

            # Smeared light propagator (backward direction)
            TO.@tensoropt (k, ) begin
                D⁻¹_light_αaβb_iₓiₓ′[α, a, β, b] :=
                    conj(v_sink_ck_iₓ′t)[b, l] * τv_light_kαβa[l, α, β, a]
            end
            #= TO.@tensoropt (k, l) begin
                D⁻¹_light_αaβb_iₓiₓ′_[α, a, β, b] :=
                    conj(v_sink_ck_iₓ′t)[b, l] *
                    γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * v_src_ck_iₓt₀[a, k]
            end
            @assert D⁻¹_light_αaβb_iₓiₓ′ ≈ D⁻¹_light_αaβb_iₓiₓ′_ =#

            # Light part
            C_light_dd′ee′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_2)
            for m in 1:Nᵧ_2
                C_light_dd′ee′_m = @view C_light_dd′ee′n[:, :, :, :, m]
                CΓ₂_αβ_m = @view CΓ₂_αβn[:, :, m]
                CΓbarC₂_αβ_m = @view CΓbarC₂_αβn[:, :, m]
                TO.@tensoropt begin
                    C_light_dd′ee′_m[d, d', e, e'] =
                        CΓ₂_αβ_m[δ, ϵ] * D⁻¹_light_αaβb_iₓiₓ′[δ', d', ϵ, e] *
                        CΓbarC₂_αβ_m[δ', ϵ'] * D⁻¹_light_αaβb_iₓiₓ′[ϵ', e', δ, d]
                end
            end

            # Positive charm part
            C_pos_charm_bb′cc′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_1)
            for n in 1:Nᵧ_1
                C_pos_charm_bb′cc′_n = @view C_pos_charm_bb′cc′n[:, :, :, :, n]
                CΓ₁_αβ_n = @view CΓ₁_αβn[:, :, n]
                CΓbarC₁_αβ_n = @view CΓbarC₁_αβn[:, :, n]
                TO.@tensoropt begin
                    C_pos_charm_bb′cc′_n[b, b', c, c'] =
                        CΓ₁_αβ_n[β, γ] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, β', b'] *
                        CΓbarC₁_αβ_n[β', γ'] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, γ', c']
                end
            end

            # Negative charm part
            C_neg_charm_bb′cc′n = Array{ComplexF64}(undef, N_c, N_c, N_c, N_c, Nᵧ_1)
            for n in 1:Nᵧ_1
                C_neg_charm_bb′cc′_n = @view C_neg_charm_bb′cc′n[:, :, :, :, n]
                CΓ₁_αβ_n = @view CΓ₁_αβn[:, :, n]
                CΓbarC₁_αβ_n = @view CΓbarC₁_αβn[:, :, n]
                TO.@tensoropt begin
                    C_neg_charm_bb′cc′_n[b, b', c, c'] =
                        CΓ₁_αβ_n[β, γ] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, γ', c'] *
                        CΓbarC₁_αβ_n[β', γ'] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, β', b']
                end
            end

            # Combine light and charm parts (sum over epsilon tensors)
            TO.@tensoropt begin
                C_nm[n, m] :=
                    (
                        C_light_dd′ee′n[b, b', c, c', m] +
                        C_light_dd′ee′n[c, c', b, b', m] -
                        C_light_dd′ee′n[b, c', c, b', m] -
                        C_light_dd′ee′n[c, b', b, c', m]
                    ) *
                    (
                        C_pos_charm_bb′cc′n[b, b', c, c', n] -
                        C_neg_charm_bb′cc′n[b, b', c, c', n]
                    )
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓ_t[:, iₓ′] - x_src_μiₓ_t₀[:, iₓ])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                C_nm_iₚ = @view C_nmiₚ[:, :, iₚ]
                TO.@tensoropt begin
                    C_nm_iₚ[n, m] += exp_mipΔx * C_nm[n, m]
                end
            end
        end
    end

    # Normalization
    C_nmiₚ .*= (prod(parms.Nₖ)/N_points)^2

    return C_nmiₚ
end