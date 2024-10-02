@doc raw"""
    DD_dad_nonlocal_local_mixed_contractons!(Cₙₗ_tnmn̄m̄iₚ::AbstractArray, Cₗₙ_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ₁_local_arr::AbstractVector{<:AbstractMatrix}, Γ₂_local_arr::AbstractVector{<:AbstractMatrix}, Γ_nonlocal_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, Iₚ_nonlocal::AbstractVector{<:Integer}, p_local_arr::AbstractVector{<:AbstractVector})

Contract the charm perambulator `τ_charm_αkβlt`, the light perambulator `τ_light_αkβlt`,
the mode doublets `Φ_kltiₚ` and the sparse Laplace modes in `sparse_modes_arrays` to
get the mixed diquark-antidiquark-DD local-nonlocal correlators. The matrices in
`Γ₁_local_arr`, `Γ₂_local_arr` and `Γ_nonlocal_arr` are the matrices that are used in the
interpolating operators. This gives vacuum expectation values of the form
(in position space) \
`ε\_{a'b'c'} ε\_{a'd'e'} \
<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄\_b' C ΓbarC₃ c̄\_c'^T  d\_d'^T C ΓbarC₄ u\_e')(x)>` \
(nonlocal-local) and \
`ε\_{abc} ε\_{ade} \
<(c\_b^T CΓ₁ c\_c  ̄u\_d CΓ₂ d̄\_e^T)(x) (c̄Γbar₃u)(x₁) (c̄Γbar₄d)(x₂)>` \
(local-nonlocal) where `Γbarᵢ = γ₄ Γᵢ^† γ₄` and `ΓbarCᵢ = Cγ₄ Γᵢ^† γ₄C`.
The gamma matrices `Γᵢ` are choose the following way: For the local operators we choose
`Γ₁, Γ₃ ∈ Γ₁_local_arr` and `Γ₂ Γ₄ ∈ Γ₂_local_arr`. For the nonlocal operators we choose all
matrices from `Γ_nonlocal_arr`. This is done for all possible combinations of them.

These two expectation values are stored in `Cₙₗ_tnmn̄m̄iₚ` (nonlocal-local) and in
`Cₗₙ_tnmn̄m̄iₚ` (local-nonlocal) where the indices n, m, n̄, m̄ correspond to the indices of the Γ's in the expectation values in the given order.

The source time `t₀` is used to circularly shift the correlators such that it is at the
origin. The two momentum indices in `Iₚ_nonlocal` are used for the momentum projection in
the nonlocal operator (positions x₁ and x₂). The array `p_local_arr` contains the integer
momenta for the momentum projection in the local operator (index iₚ in `Cₙₗ_tnmn̄m̄iₚ` and
`Cₗₙ_tnmn̄m̄iₚ`).
"""
function DD_dad_nonlocal_local_mixed_contractons!(
    Cₙₗ_tnmn̄m̄iₚ::AbstractArray, Cₗₙ_tnmn̄m̄iₚ::AbstractArray,
    τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    Φ_kltiₚ::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray},
    Γ₁_local_arr::AbstractVector{<:AbstractMatrix},
    Γ₂_local_arr::AbstractVector{<:AbstractMatrix},
    Γ_nonlocal_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer,
    Iₚ_nonlocal::AbstractVector{<:Integer}, p_local_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    _, N_points, _ = size(x_sink_μiₓt)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    # for the nonlocal operators
    Γₙₗ_αβn = stack(Γ_nonlocal_arr)
    TO.@tensoropt Γbarₙₗ_αβn[α, β, n] := γ[4][α, α'] * conj(Γₙₗ_αβn)[β', α', n] * γ[4][β', β]
    # and the local operators
    Γₗ₁_αβn = stack(Γ₁_local_arr)
    Γₗ₂_αβn = stack(Γ₂_local_arr)
    TO.@tensoropt CΓₗ₁_αβn[α, β, n] := C[α, α'] * Γₗ₁_αβn[α', β, n]
    TO.@tensoropt CΓₗ₂_αβn[α, β, n] := C[α, α'] * Γₗ₂_αβn[α', β, n]
    TO.@tensoropt CΓbarCₗ₁_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γₗ₁_αβn)[β', α'', n] * γ[2][β', β]
    TO.@tensoropt CΓbarCₗ₂_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γₗ₂_αβn)[β', α'', n] * γ[2][β', β]

    # Convert momentum array to contiguous array
    p_local_μiₚ = stack(p_local_arr)

    # Set correlators to zero and permute such that time is slowest changing index
    Cₙₗ_tnmn̄m̄iₚ .= 0
    Cₗₙ_tnmn̄m̄iₚ .= 0
    Cₙₗ_nmn̄m̄iₚt = permutedims(Cₙₗ_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])
    Cₗₙ_nmn̄m̄iₚt = permutedims(Cₗₙ_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])

    # Mode doublet at source time `t₀` for momenta `Iₚ_nonlocal[1]` and `Iₚ_nonlocal[2]`
    Φ_kl_t₀p₁ = @view Φ_kltiₚ[:, :, i_t₀, Iₚ_nonlocal[1]]
    Φ_kl_t₀p₂ = @view Φ_kltiₚ[:, :, i_t₀, Iₚ_nonlocal[2]]

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Time index for storing correlator entrie
        i_Δt = mod1(iₜ-t₀, parms.Nₜ)

        # Perambulators at sink time t
        τ_charm_αkβl_t = @view τ_charm_αkβlt[:, :, :, :, iₜ]
        τ_light_αkβl_t = @view τ_light_αkβlt[:, :, :, :, iₜ]

        # Light perambulator in backward direction
        TO.@tensoropt (l, k) begin
            τ_bw_light_kαβl_t[k, α, β, l] :=
                γ[5][β, β'] * conj(τ_light_αkβl_t)[β', l, α', k] * γ[5][α', α]
        end

        # Mode doublets at sink time t for momenta `Iₚ_nonlocal[1]` and `Iₚ_nonlocal[2]`
        Φ_kl_tp₁ = @view Φ_kltiₚ[:, :, iₜ, Iₚ_nonlocal[1]]
        Φ_kl_tp₂ = @view Φ_kltiₚ[:, :, iₜ, Iₚ_nonlocal[2]]

        # Nonlocal-local tensor contractions
        ####################################

        # Pre-contractions
        TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            A[k, α', β', l'] := Φ_kl_tp₁[k, k'] * τ_charm_αkβl_t[α', k', β', l']
            B[l, α_, β, k̃'] := τ_charm_αkβl_t[α_, k̃, β, l] * Φ_kl_tp₂[k̃', k̃]
        end

        # Loop over source position iₓ
        for iₓ in 1:N_points
            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Pre-contractions
            TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C1[α', β', α, β_, b, e] :=
                    v_src_ck_iₓt₀[e, l̃] * τ_bw_light_kαβl_t[l̃, β_, α, k] *
                    A[k, α', β', l'] * conj(v_src_ck_iₓt₀)[b, l']
                
                C2[α_, β, α_', β_', c, d] := 
                    conj(v_src_ck_iₓt₀)[c, l] * B[l, α_, β, k̃'] *
                    τ_bw_light_kαβl_t[l̃', β_', α_', k̃'] * v_src_ck_iₓt₀[d, l̃']
            end

            # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    Γₙₗ_αβn[α, α', n] * C1[α', β', α, β_, b, e] *
                    CΓbarCₗ₁_αβn[β', β, n̄] * Γₙₗ_αβn[α_', α_, m] *
                    C2[α_, β, α_', β_', c, d] * CΓbarCₗ₂_αβn[β_', β_, m̄]
            end
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_pos_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    Φ_kl_tp₁[k, k'] * Γₙₗ_αβn[α, α', n] *
                    τ_charm_αkβl_t[α', k', β', l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * conj(v_src_ck_iₓt₀)[c, l] *
                    CΓbarCₗ₁_αβn[β', β, n̄] * τ_charm_αkβl_t[α_, k̃, β, l] *
                    Φ_kl_tp₂[k̃', k̃] * Γₙₗ_αβn[α_', α_, m] *
                    τ_bw_light_kαβl_t[l̃', β_', α_', k̃'] *
                    v_src_ck_iₓt₀[d, l̃'] * v_src_ck_iₓt₀[e, l̃] *
                    CΓbarCₗ₂_αβn[β_', β_, m̄] * τ_bw_light_kαβl_t[l̃, β_, α, k]
            end
            @assert C_pos_bcdenmn̄m̄ ≈ C_pos_bcdenmn̄m̄_old =#

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    Γₙₗ_αβn[α, α', n] * C1[α', β', α, β_, c, e] *
                    CΓbarCₗ₁_αβn[β, β', n̄] * Γₙₗ_αβn[α_', α_, m] *
                    C2[α_, β, α_', β_', b, d] * CΓbarCₗ₂_αβn[β_', β_, m̄]
            end
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_neg_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    Φ_kl_tp₁[k, k'] * Γₙₗ_αβn[α, α', n] *
                    τ_charm_αkβl_t[α', k', β', l'] *
                    conj(v_src_ck_iₓt₀)[c, l'] * conj(v_src_ck_iₓt₀)[b, l] *
                    CΓbarCₗ₁_αβn[β, β', n̄] * τ_charm_αkβl_t[α_, k̃, β, l] *
                    Φ_kl_tp₂[k̃', k̃] * Γₙₗ_αβn[α_', α_, m] *
                    τ_bw_light_kαβl_t[l̃', β_', α_', k̃'] *
                    v_src_ck_iₓt₀[d, l̃'] * v_src_ck_iₓt₀[e, l̃] *
                    CΓbarCₗ₂_αβn[β_', β_, m̄] * τ_bw_light_kαβl_t[l̃, β_, α, k]
            end
            @assert C_neg_bcdenmn̄m̄ ≈ C_neg_bcdenmn̄m̄_old =#

            # Sum over epsilon tensors
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_pos_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_neg_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_pos_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄] +
                    C_neg_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄]
            end

            # Momentum projection
            p2πix = 2π*im * 
                x_src_μiₓt[:, iₓ, i_t₀]./parms.Nₖ
            exp_ipx_arr = exp.(p_local_μiₚ' * p2πix)
            for iₚ in eachindex(p_local_arr)
                # Use Δt=t-t₀ as time
                Cₙₗ_nmn̄m̄_iₚΔt = @view Cₙₗ_nmn̄m̄iₚt[:, :, :, :, iₚ, i_Δt]

                exp_ipx = exp_ipx_arr[iₚ]
                TO.@tensoropt begin
                    Cₙₗ_nmn̄m̄_iₚΔt[n, m, n̄, m̄] += exp_ipx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end

        # Local-nonlocal tensor contractions
        ####################################

        # Pre-contractions
        TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            A[k', l, α', β'] := τ_charm_αkβl_t[α', k', β', l'] * conj(Φ_kl_t₀p₁)[l, l']
            B[k̃', l̃, α_', β_'] :=
                τ_bw_light_kαβl_t[l̃', β_' , α_', k̃'] * conj(Φ_kl_t₀p₂)[l̃', l̃]
        end

        # Loop over sink position iₓ′
        for iₓ′ in 1:N_points
            # Laplace modes at sink time t and position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

            # Pre-contractions
            TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C1[α', β', α_, β, c, d] :=
                    v_sink_ck_iₓ′t[c, k'] * A[k', l, α', β'] * 
                    τ_bw_light_kαβl_t[l, β, α_, k̃] * conj(v_sink_ck_iₓ′t)[d, k̃]
                
                C2[α_', β_', α, β_, b, e] := 
                    conj(v_sink_ck_iₓ′t)[e, k̃'] * B[k̃', l̃, α_', β_'] *
                    τ_charm_αkβl_t[α, k, β_, l̃] * v_sink_ck_iₓ′t[b, k]
            end

            # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    C1[α', β', α_, β, c, d] * CΓₗ₁_αβn[α, α', n] *
                    Γbarₙₗ_αβn[β', β, n̄] * C2[α_', β_', α, β_, b, e] *
                    CΓₗ₂_αβn[α_, α_', m] * Γbarₙₗ_αβn[β_, β_', m̄]
            end
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_pos_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    v_sink_ck_iₓ′t[b, k] * v_sink_ck_iₓ′t[c, k'] *
                    CΓₗ₁_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β', l'] *
                    conj(Φ_kl_t₀p₁)[l, l'] * Γbarₙₗ_αβn[β', β, n̄] *
                    τ_bw_light_kαβl_t[l, β, α_, k̃] *
                    conj(v_sink_ck_iₓ′t)[d, k̃] * conj(v_sink_ck_iₓ′t)[e, k̃'] *
                    CΓₗ₂_αβn[α_, α_', m] * τ_bw_light_kαβl_t[l̃', β_' , α_', k̃'] *
                    conj(Φ_kl_t₀p₂)[l̃', l̃] * Γbarₙₗ_αβn[β_, β_', m̄] *
                    τ_charm_αkβl_t[α, k, β_, l̃]
            end
            @assert C_pos_bcdenmn̄m̄ ≈ C_pos_bcdenmn̄m̄_old =#

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    C1[α', β', α_, β, b, d] * CΓₗ₁_αβn[α', α, n] *
                    Γbarₙₗ_αβn[β', β, n̄] * C2[α_', β_', α, β_, c, e] *
                    CΓₗ₂_αβn[α_, α_', m] * Γbarₙₗ_αβn[β_, β_', m̄]
            end
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_neg_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    v_sink_ck_iₓ′t[c, k] * v_sink_ck_iₓ′t[b, k'] *
                    CΓₗ₁_αβn[α', α, n] * τ_charm_αkβl_t[α', k', β', l'] *
                    conj(Φ_kl_t₀p₁)[l, l'] * Γbarₙₗ_αβn[β', β, n̄] *
                    τ_bw_light_kαβl_t[l, β, α_, k̃] *
                    conj(v_sink_ck_iₓ′t)[d, k̃] * conj(v_sink_ck_iₓ′t)[e, k̃'] *
                    CΓₗ₂_αβn[α_, α_', m] * τ_bw_light_kαβl_t[l̃', β_' , α_', k̃'] *
                    conj(Φ_kl_t₀p₂)[l̃', l̃] * Γbarₙₗ_αβn[β_, β_', m̄] *
                    τ_charm_αkβl_t[α, k, β_, l̃]
            end
            @assert C_neg_bcdenmn̄m̄ ≈ C_neg_bcdenmn̄m̄_old =#

            # Sum over epsilon tensors
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_pos_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_neg_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_pos_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄] +
                    C_neg_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄]
            end

            # Momentum projection
            m2πix = -2π*im * 
                x_sink_μiₓt[:, iₓ′, iₜ]./parms.Nₖ
            exp_ipx_arr = exp.(p_local_μiₚ' * m2πix)
            for iₚ in eachindex(p_local_arr)
                # Use Δt=t-t₀ as time
                Cₗₙ_nmn̄m̄_iₚΔt = @view Cₗₙ_nmn̄m̄iₚt[:, :, :, :, iₚ, i_Δt]

                exp_ipx = exp_ipx_arr[iₚ]
                TO.@tensoropt begin
                    Cₗₙ_nmn̄m̄_iₚΔt[n, m, n̄, m̄] += exp_ipx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end
    end

    # Permute correlator back
    permutedims!(Cₙₗ_tnmn̄m̄iₚ, Cₙₗ_nmn̄m̄iₚt, [6, 1, 2, 3, 4, 5])
    permutedims!(Cₗₙ_tnmn̄m̄iₚ, Cₗₙ_nmn̄m̄iₚt, [6, 1, 2, 3, 4, 5])

    # Normalization
    Cₙₗ_tnmn̄m̄iₚ *= prod(parms.Nₖ)/N_points
    Cₗₙ_tnmn̄m̄iₚ *= prod(parms.Nₖ)/N_points

    return
end

@doc raw"""
    DD_dad_local_mixed_contractons!(C_DD_dad_tnmn̄m̄iₚ::AbstractArray, C_dad_DD_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ₁_dad_arr::AbstractVector{<:AbstractMatrix}, Γ₂_dad_arr::AbstractVector{<:AbstractMatrix}, Γ_DD_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, p_arr::AbstractVector{<:AbstractVector})

Contract the charm perambulator `τ_charm_αkβlt` and the light perambulator `τ_light_αkβlt`
and the sparse Laplace modes in `sparse_modes_arrays` to get the mixed
diquark-antidiquark-DD (dad-DD) local correlators.
The matrices in `Γ₁_dad_arr`, `Γ₂_dad_arr` and `Γ_DD_arr` are the matrices in
the interpolating operators. This gives vacuum expectation values of the form
(in position space) \
`ε\_{a'b'c'} ε\_{a'd'e'} \
<(ūΓ₁c d̄Γ₂c)(x') (c̄\_b' C ΓbarC₃ c̄\_c'^T  d\_d'^T C ΓbarC₄ u\_e')(x)>` \
(DD-dad) and \
`ε\_{abc} ε\_{ade} \
<(c\_b^T CΓ₁ c\_c  ̄u\_d CΓ₂ d̄\_e^T)(x') (c̄Γbar₃u c̄Γbar₄d)(x)>` \
(dad-DD) where `Γbarᵢ = γ₄ Γᵢ^† γ₄` and `ΓbarCᵢ = Cγ₄ Γᵢ^† γ₄C`. 
The gamma matrices `Γᵢ` are choose the following way: For the dad operators we choose
`Γ₁, Γ₃ ∈ Γ₁_dad_arr` and `Γ₂ Γ₄ ∈ Γ₂_dad_arr`. For the DD operators we choose all matrices
from `Γ_DD_arr`. This is done for all possible combinations of them.

These two expectation values are stored in `C_DD_dad_tnmn̄m̄iₚ` (DD-diquark-antidiquark) and
in `C_dad_DD_tnmn̄m̄iₚ` (diquark-antidiquark-DD) where the indices n, m, n̄, m̄ correspond to
the indices of the Γ's in the expectation values in the given order.

The source time `t₀` is used to circularly shift the correlators such that it is at
the origin. The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in `C_DD_dad_tnmn̄m̄iₚ` and `C_dad_DD_tnmn̄m̄iₚ`).
"""
function DD_dad_local_mixed_contractons!(
    C_DD_dad_tnmn̄m̄iₚ::AbstractArray, C_dad_DD_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray},
    Γ₁_dad_arr::AbstractVector{<:AbstractMatrix},
    Γ₂_dad_arr::AbstractVector{<:AbstractMatrix},
    Γ_DD_arr::AbstractVector{<:AbstractMatrix},
    t₀::Integer, p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓt, 2)

    # Number of colors (could probably just use 3)
    N_c = size(v_sink_ciₓkt, 1)

    # Convert vector of γ-matrices to contiguous array and compute relevant matrices
    # for the DD operators
    Γ_DD_αβn = stack(Γ_DD_arr)
    TO.@tensoropt Γbar_DD_αβn[α, β, n] :=
        γ[4][α, α'] * conj(Γ_DD_αβn)[β', α', n] * γ[4][β', β]
    # and the dad operators
    Γ₁_dad_αβn = stack(Γ₁_dad_arr)
    Γ₂_dad_αβn = stack(Γ₂_dad_arr)
    TO.@tensoropt CΓ₁_dad_αβn[α, β, n] := C[α, α'] * Γ₁_dad_αβn[α', β, n]
    TO.@tensoropt CΓ₂_dad_αβn[α, β, n] := C[α, α'] * Γ₂_dad_αβn[α', β, n]
    TO.@tensoropt CΓbarC₁_dad_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₁_dad_αβn)[β', α'', n] * γ[2][β', β]
    TO.@tensoropt CΓbarC₂_dad_αβn[α, β, n] :=
        C[α, α'] * γ[2][α', α''] * conj(Γ₂_dad_αβn)[β', α'', n] * γ[2][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Set correlators to zero and permute such that time is slowest changing index
    C_DD_dad_tnmn̄m̄iₚ .= 0
    C_dad_DD_tnmn̄m̄iₚ .= 0
    C_DD_dad_nmn̄m̄iₚt = permutedims(C_DD_dad_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])
    C_dad_DD_nmn̄m̄iₚt = permutedims(C_dad_DD_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Time index for storing correlator entrie
        i_Δt = mod1(iₜ-t₀, parms.Nₜ)

        # Perambulators at sink time t
        τ_charm_αkβl_t = @view τ_charm_αkβlt[:, :, :, :, iₜ]
        τ_light_αkβl_t = @view τ_light_αkβlt[:, :, :, :, iₜ]

        # Light perambulator in backward direction
        TO.@tensoropt (l, k) begin
            τ_bw_light_kαβl_t[k, α, β, l] :=
                γ[5][β, β'] * conj(τ_light_αkβl_t)[β', l, α', k] * γ[5][α', α]
        end

        # Loop over sink position iₓ′ and source position iₓ
        for iₓ′ in 1:N_points, iₓ in 1:N_points
            # Laplace modes at sink time t and position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Smeared charm propagator (forward direction)
            TO.@tensoropt (k, l) begin
                D⁻¹_charm_αaβb_iₓ′iₓ[α, a, β, b] :=
                    v_sink_ck_iₓ′t[a, k] * 
                    τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
            end

            # Smeared light propagator (backward direction)
            TO.@tensoropt (k, l) begin
                D⁻¹_light_αaβb_iₓiₓ′[α, a, β, b] :=
                    v_src_ck_iₓt₀[a, k] *
                    τ_bw_light_kαβl_t[k, α, β, l] * conj(v_sink_ck_iₓ′t)[b, l]
            end

            # DD-dad tensor contractions
            ############################

            # Pre-contractions
            TO.@tensoropt begin
                A1[β, β_', c', d', m] :=
                    D⁻¹_charm_αaβb_iₓ′iₓ[α_, b, β, c'] * Γ_DD_αβn[α_', α_, m] *
                    D⁻¹_light_αaβb_iₓiₓ′[β_', d', α_', b]
                A2[β_', β', b', e', n, m̄] :=
                    CΓbarC₂_dad_αβn[β_', β_, m̄] * D⁻¹_light_αaβb_iₓiₓ′[β_, e', α, a] *
                    Γ_DD_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β', b']
            end            

            # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄[b', c', d', e', n, m, n̄, m̄] :=
                    CΓbarC₁_dad_αβn[β', β, n̄] *
                    A1[β, β_', c', d', m] * A2[β_', β', b', e', n, m̄]
            end

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄[b', c', d', e', n, m, n̄, m̄] :=
                    CΓbarC₁_dad_αβn[β, β', n̄] *
                    A1[β, β_', b', d', m] * A2[β_', β', c', e', n, m̄]
            end

            #= # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄_old[b', c', d', e', n, m, n̄, m̄] :=
                    Γ_DD_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β', b'] *
                    CΓbarC₁_dad_αβn[β', β, n̄] * D⁻¹_charm_αaβb_iₓ′iₓ[α_, b, β, c'] *
                    Γ_DD_αβn[α_', α_, m] * D⁻¹_light_αaβb_iₓiₓ′[β_', d', α_', b] *
                    CΓbarC₂_dad_αβn[β_', β_, m̄] * D⁻¹_light_αaβb_iₓiₓ′[β_, e', α, a]
            end
            @assert C_pos_bcdenmn̄m̄ ≈ C_pos_bcdenmn̄m̄_old

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄_old[b', c', d', e', n, m, n̄, m̄] :=
                    Γ_DD_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β', c'] *
                    CΓbarC₁_dad_αβn[β, β', n̄] * D⁻¹_charm_αaβb_iₓ′iₓ[α_, b, β, b'] *
                    Γ_DD_αβn[α_', α_, m] * D⁻¹_light_αaβb_iₓiₓ′[β_', d', α_', b] *
                    CΓbarC₂_dad_αβn[β_', β_, m̄] * D⁻¹_light_αaβb_iₓiₓ′[β_, e', α, a]
            end
            @assert C_neg_bcdenmn̄m̄ ≈ C_neg_bcdenmn̄m̄_old =#

            # Sum over epsilon tensors
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_pos_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_neg_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_pos_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄] +
                    C_neg_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄]
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓt[:, iₓ′, iₜ] - x_src_μiₓt[:, iₓ, i_t₀])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                # Use Δt=t-t₀ as time
                C_DD_dad_nmn̄m̄_iₚΔt = @view C_DD_dad_nmn̄m̄iₚt[:, :, :, :, iₚ, i_Δt]
                TO.@tensoropt begin
                    C_DD_dad_nmn̄m̄_iₚΔt[n, m, n̄, m̄] += exp_mipΔx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end

            # dad-DD tensor contractions
            ############################

            # Pre-contractions
            TO.@tensoropt begin
                A1[β, β_', c, d, n̄] :=
                    D⁻¹_charm_αaβb_iₓ′iₓ[β, c, α_, a'] * Γbar_DD_αβn[α_, α_', n̄] * D⁻¹_light_αaβb_iₓiₓ′[α_', a', β_', d]
                A2[β_', β', b, e, m, m̄] :=
                    CΓ₂_dad_αβn[β_', β_, m] * D⁻¹_light_αaβb_iₓiₓ′[α, b', β_, e] *
                    Γbar_DD_αβn[α', α, m̄] * D⁻¹_charm_αaβb_iₓ′iₓ[β', b, α', b']
            end
            
            # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    CΓ₁_dad_αβn[β', β, n] *
                    A1[β, β_', c, d, n̄] * A2[β_', β', b, e, m, m̄]
            end

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄[b, c, d, e, n, m, n̄, m̄] :=
                    CΓ₁_dad_αβn[β, β', n] *
                    A1[β, β_', b, d, n̄] * A2[β_', β', c, e, m, m̄]
            end

            #= # Positive part
            TO.@tensoropt begin
                C_pos_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    Γbar_DD_αβn[α', α, m̄] * D⁻¹_charm_αaβb_iₓ′iₓ[β', b, α', b'] *
                    CΓ₁_dad_αβn[β', β, n] * D⁻¹_charm_αaβb_iₓ′iₓ[β, c, α_, a'] *
                    Γbar_DD_αβn[α_, α_', n̄] * D⁻¹_light_αaβb_iₓiₓ′[α_', a', β_', d] *
                    CΓ₂_dad_αβn[β_', β_, m] * D⁻¹_light_αaβb_iₓiₓ′[α, b', β_, e]
            end
            @assert C_pos_bcdenmn̄m̄ ≈ C_pos_bcdenmn̄m̄_old

            # Negative part
            TO.@tensoropt begin
                C_neg_bcdenmn̄m̄_old[b, c, d, e, n, m, n̄, m̄] :=
                    Γbar_DD_αβn[α', α, m̄] * D⁻¹_charm_αaβb_iₓ′iₓ[β', c, α', b'] *
                    CΓ₁_dad_αβn[β, β', n] * D⁻¹_charm_αaβb_iₓ′iₓ[β, b, α_, a'] *
                    Γbar_DD_αβn[α_, α_', n̄] * D⁻¹_light_αaβb_iₓiₓ′[α_', a', β_', d] *
                    CΓ₂_dad_αβn[β_', β_, m] * D⁻¹_light_αaβb_iₓiₓ′[α, b', β_, e]
            end
            @assert C_neg_bcdenmn̄m̄ ≈ C_neg_bcdenmn̄m̄_old =#

            # Sum over epsilon tensors
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_pos_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_neg_bcdenmn̄m̄[b, c, b, c, n, m, n̄, m̄] -
                    C_pos_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄] +
                    C_neg_bcdenmn̄m̄[c, b, b, c, n, m, n̄, m̄]
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓt[:, iₓ′, iₜ] - x_src_μiₓt[:, iₓ, i_t₀])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                # Use Δt=t-t₀ as time
                C_dad_DD_nmn̄m̄_iₚΔt = @view C_dad_DD_nmn̄m̄iₚt[:, :, :, :, iₚ, i_Δt]
                TO.@tensoropt begin
                    C_dad_DD_nmn̄m̄_iₚΔt[n, m, n̄, m̄] += exp_mipΔx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end
    end

    # Permute correlator back
    permutedims!(C_DD_dad_tnmn̄m̄iₚ, C_DD_dad_nmn̄m̄iₚt, [6, 1, 2, 3, 4, 5])
    permutedims!(C_dad_DD_tnmn̄m̄iₚ, C_dad_DD_nmn̄m̄iₚt, [6, 1, 2, 3, 4, 5])

    # Normalization
    C_DD_dad_tnmn̄m̄iₚ *= (prod(parms.Nₖ)/N_points)^2
    C_dad_DD_tnmn̄m̄iₚ *= (prod(parms.Nₖ)/N_points)^2

    return
end
