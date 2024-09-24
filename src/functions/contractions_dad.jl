@doc raw"""
    dad_local_contractons!(C_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, p_arr::AbstractVector{<:AbstractVector}

Contract the charm perambulator `τ_charm_αkβlt` and the light perambulator `τ_light_αkβlt`
and the sparse Laplace modes in `sparse_modes_arrays` to get the local diquark-antidiquark
correlator. The matrices in `Γ_arr` are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum
expectation value of the form \
`ε\_{abc} ε\_{ade} ε\_{a'b'c'} ε\_{a'd'e'} \
<(c\_b^T CΓ₁ c\_c  ̄u\_d CΓ₂ d̄\_e^T)(x')
(c̄\_b' C ΓbarC₃ c̄\_c'^T  d\_d'^T C ΓbarC₄ u\_e')(x)>` \
(in position space) where `ΓbarCᵢ = Cγ₄ Γᵢ^† γ₄C`. The result is stored in `C_tnmn̄m̄iₚ`
where the indices n, m, n̄, m̄ correspond to the indices of the Γ's in the expectation value
in the given order.

The source time `t₀` is used to circularly shift `C_tnmn̄m̄iₚ` such that it is at
the origin. The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in `C_tnmn̄m̄iₚ`).
"""
function dad_local_contractons!(
    C_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix},
    t₀::Integer, p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓt, 2)

    # Convert vector of γ-matrices to contiguous array and compute CΓ and CΓbarC matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt CΓ_αβn[α, β, n] := C[α, α'] * Γ_αβn[α', β, n]
    TO.@tensoropt CΓbarC_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ_αβn)[β', α'', n] * γ[2][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Set correlator C_tnmn̄m̄iₚ to zero and permute such that time is slowest changing index
    C_tnmn̄m̄iₚ .= 0
    C_nmn̄m̄iₚt = permutedims(C_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])

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

        # Loop over sink position iₓ′ and source position iₓ
        for iₓ′ in 1:N_points
        for iₓ in 1:N_points
            # Laplace modes at sink time t and position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Tensor contractions
            #####################

            # Smeared charm propagator (forward direction)
            TO.@tensoropt (k, l) begin
                D⁻¹_charm_αaβb_iₓ′iₓ[α, a, β, b] :=
                    v_sink_ck_iₓ′t[a, k] * 
                    τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
            end

            # Smeared light propagator (backward direction)
            TO.@tensoropt (k, l) begin
                D⁻¹_light_αaβb_iₓiₓ′[α, a, β, b] :=
                    conj(v_sink_ck_iₓ′t)[b, l] *
                    γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * v_src_ck_iₓt₀[a, k]
            end

            # Light part
            TO.@tensoropt begin
                C_light_dd′ee′nn̄[d, d', e, e', m, m̄] :=
                    CΓ_αβn[δ, ϵ, m] * D⁻¹_light_αaβb_iₓiₓ′[δ', d', ϵ, e] *
                    CΓbarC_αβn[δ', ϵ', m̄] * D⁻¹_light_αaβb_iₓiₓ′[ϵ', e', δ, d]
            end

            # Positive charm part
            TO.@tensoropt begin
                C_pos_charm_bb′cc′nn̄[b, b', c, c', n, n̄] :=
                    CΓ_αβn[β, γ, n] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, β', b'] *
                    CΓbarC_αβn[β', γ', n̄] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, γ', c']
            end

            # Negative charm part
            TO.@tensoropt begin
                C_neg_charm_bb′cc′nn̄[b, b', c, c', n, n̄] :=
                    CΓ_αβn[β, γ, n] * D⁻¹_charm_αaβb_iₓ′iₓ[γ, c, γ', c'] *
                    CΓbarC_αβn[β', γ', n̄] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, β', b']
            end

            # Combine light and charm parts (sum over epsilon tensors)
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    (
                        C_light_dd′ee′nn̄[b, b', c, c', m, m̄] +
                        C_light_dd′ee′nn̄[c, c', b, b', m, m̄] -
                        C_light_dd′ee′nn̄[b, c', c, b', m, m̄] -
                        C_light_dd′ee′nn̄[c, b', b, c', m, m̄]
                    ) *
                    (
                        C_pos_charm_bb′cc′nn̄[b, b', c, c', n, n̄] -
                        C_neg_charm_bb′cc′nn̄[b, b', c, c', n, n̄]
                    )
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓt[:, iₓ′, iₜ] - x_src_μiₓt[:, iₓ, i_t₀])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                # Use Δt=t-t₀ as time
                C_nmn̄m̄_iₚΔt = @view C_nmn̄m̄iₚt[:, :, :, :, iₚ, i_Δt]
                TO.@tensoropt begin
                    C_nmn̄m̄_iₚΔt[n, m, n̄, m̄] += exp_mipΔx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end
        end
    end

    # Permute correlator back
    permutedims!(C_tnmn̄m̄iₚ, C_nmn̄m̄iₚt, [6, 1, 2, 3, 4, 5])

    # Normalization
    C_tnmn̄m̄iₚ *= (prod(parms.Nₖ)/N_points)^2

    return
end
