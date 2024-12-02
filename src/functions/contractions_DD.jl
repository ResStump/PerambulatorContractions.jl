@doc raw"""
    DD_local_contractons!(C_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, p_arr::AbstractVector{<:AbstractVector})

Contract the charm perambulator `τ_charm_αkβlt` and the light perambulator `τ_light_αkβlt`
and the sparse Laplace modes in `sparse_modes_arrays` to get the local DD
correlator. The matrices in `Γ_arr` are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum
expectation value of the form \
`<(ūΓ₁c d̄Γ₂c)(x') (c̄Γbar₃u c̄Γbar₄d)(x)>` \
(in position space). The result is stored in `C_tnmn̄m̄iₚ` where the indices n, m, n̄, m̄
correspond to the indices of the Γ's in the expectation value in the given order.

The source time `t₀` is used to circularly shift `C_tnmn̄m̄iₚ` such that it is at
the origin. The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in `C_tnmn̄m̄iₚ`).
"""
function DD_local_contractons!(
    C_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix},
    t₀::Integer, p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    _, N_points, _ = size(x_sink_μiₓt)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Set correlator C_tnmn̄m̄iₚ to zero and permute such that time is slowest changing index
    C_tnmn̄m̄iₚ .= 0
    C_nmn̄m̄iₚt = permutedims(C_tnmn̄m̄iₚ, [2, 3, 4, 5, 6, 1])

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Time index for storing correlator entry
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

                # Disconnected part
                #= TO.@tensoropt (l, k, l', k') begin
                    C_disc_nn̄[n, n̄] :=
                        conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                        Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                        conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                        Γbar_αβn[β, β', n̄] * γ₅τ_conjγ₅_light_αkβl_t[α, k, β', l]
                end =#
                TO.@tensoropt begin
                    C_disc_nn̄[n, n̄] :=
                        Γ_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β, b] *
                        Γbar_αβn[β, β', n̄] * D⁻¹_light_αaβb_iₓiₓ′[β', b, α, a]
                end

                # Connected part
                #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                    C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                        conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                        Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                        conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                        Γbar_αβn[β, β', m̄] * γ₅τ_conjγ₅_light_αkβl_t[α_, k̃, β', l] *
                        conj(v_sink_ck_iₓ′t)[ã, k̃] * v_sink_ck_iₓ′t[ã, k̃'] *
                        Γ_αβn[α_, α_', m] * τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                        conj(v_src_ck_iₓt₀)[b̃, l̃'] * v_src_ck_iₓt₀[b̃, l̃] *
                        Γbar_αβn[β_, β_', n̄] * γ₅τ_conjγ₅_light_αkβl_t[α, k, β_', l̃]
                end =#
                TO.@tensoropt begin
                    C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                        Γ_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β, b] *
                        Γbar_αβn[β, β', m̄] * D⁻¹_light_αaβb_iₓiₓ′[β', b, α_, ã] *
                        Γ_αβn[α_, α_', m] * D⁻¹_charm_αaβb_iₓ′iₓ[α_', ã, β_, b̃] *
                        Γbar_αβn[β_, β_', n̄] * D⁻¹_light_αaβb_iₓiₓ′[β_', b̃, α, a]
                end

                # Combine connected and disconnected part
                TO.@tensoropt begin
                    C_nmn̄m̄[n, m, n̄, m̄] :=
                        C_disc_nn̄[n, n̄]*C_disc_nn̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
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
    C_tnmn̄m̄iₚ .*= (prod(parms.Nₖ)/N_points)^2

    return
end

@doc raw"""
    DD_local_contractons(τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray, sparse_modes_arrays_tt₀::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, p_arr::AbstractVector{<:AbstractVector}) -> C_nmn̄m̄iₚ::AbstractArray

Contract the charm perambulator `τ_charm_αkβl_t` and the light perambulator `τ_light_αkβl_t`
and the sparse Laplace modes in `sparse_modes_arrays_tt₀` to get the local DD correlator and
return it. These arrays are assumed to only contain data for a single sink time `t` and
source time `t₀`. The matrices in `Γ_arr` are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum
expectation value of the form \
`<(ūΓ₁c d̄Γ₂c)(x') (c̄Γbar₃u c̄Γbar₄d)(x)>` \
(in position space). The result is returned as the array `C_nmn̄m̄iₚ` where the indices
n, m, n̄, m̄ correspond to the indices of the Γ's in the expectation value in the given order.

The array `p_arr` contains the integer momenta the correlator is projected to
(index iₚ in `C_nmn̄m̄iₚ`).
"""
function DD_local_contractons(
    τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray,
    sparse_modes_arrays_tt₀::NTuple{4, AbstractArray},
    Γ_arr::AbstractVector{<:AbstractMatrix}, p_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓ_t, x_src_μiₓ_t, v_sink_ciₓk_t, v_src_ciₓk_t₀ = sparse_modes_arrays_tt₀

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓ_t, 2)

    # Number of gamma matrices
    Nᵧ = length(Γ_arr)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

    # Convert momentum array to contiguous array
    p_μiₚ = stack(p_arr)

    # Allocate correlator
    C_nmn̄m̄iₚ = zeros(ComplexF64, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(p_arr))

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

            # Disconnected part
            #= TO.@tensoropt (l, k, l', k') begin
                C_disc_nn̄[n, n̄] :=
                    conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                    Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                    Γbar_αβn[β, β', n̄] * γ₅τ_conjγ₅_light_αkβl_t[α, k, β', l]
            end =#
            TO.@tensoropt begin
                C_disc_nn̄[n, n̄] :=
                    Γ_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β, b] *
                    Γbar_αβn[β, β', n̄] * D⁻¹_light_αaβb_iₓiₓ′[β', b, α, a]
            end

            # Connected part
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                    Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                    Γbar_αβn[β, β', m̄] * γ₅τ_conjγ₅_light_αkβl_t[α_, k̃, β', l] *
                    conj(v_sink_ck_iₓ′t)[ã, k̃] * v_sink_ck_iₓ′t[ã, k̃'] *
                    Γ_αβn[α_, α_', m] * τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                    conj(v_src_ck_iₓt₀)[b̃, l̃'] * v_src_ck_iₓt₀[b̃, l̃] *
                    Γbar_αβn[β_, β_', n̄] * γ₅τ_conjγ₅_light_αkβl_t[α, k, β_', l̃]
            end =#
            TO.@tensoropt begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    Γ_αβn[α, α', n] * D⁻¹_charm_αaβb_iₓ′iₓ[α', a, β, b] *
                    Γbar_αβn[β, β', m̄] * D⁻¹_light_αaβb_iₓiₓ′[β', b, α_, ã] *
                    Γ_αβn[α_, α_', m] * D⁻¹_charm_αaβb_iₓ′iₓ[α_', ã, β_, b̃] *
                    Γbar_αβn[β_, β_', n̄] * D⁻¹_light_αaβb_iₓiₓ′[β_', b̃, α, a]
            end

            # Combine connected and disconnected part
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_disc_nn̄[n, n̄]*C_disc_nn̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓ_t[:, iₓ′] - x_src_μiₓ_t[:, iₓ])./parms.Nₖ
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                C_nmn̄m̄_iₚ = @view C_nmn̄m̄iₚ[:, :, :, :, iₚ]
                TO.@tensoropt begin
                    C_nmn̄m̄_iₚ[n, m, n̄, m̄] += exp_mipΔx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end
    end

    # Normalization
    C_nmn̄m̄iₚ .*= (prod(parms.Nₖ)/N_points)^2

    return C_nmn̄m̄iₚ
end

@doc raw"""
    DD_nonlocal_contractons!(C_tnmn̄m̄::AbstractArray, τ_charm_αkβlt::AbstractArray,  τ_light_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray, Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, Iₚ::AbstractVector{<:Integer}; swap_ud::Bool=false)

Contract the charm perambulator `τ_charm_αkβlt`, the light perambulator `τ_light_αkβlt`
and the mode doublets `Φ_kltiₚ` to get the nolocal DD correlator. The matrices in `Γ_arr`
are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum
expectation value of the form (in position space) \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃u)(x₃) (c̄Γbar₄d)(x₄)>` \
 or \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃d)(x₃) (c̄Γbar₄u)(x₄)>` \
if `swap_ud` = true.
    
The result is stored in `C_tnmn̄m̄` where the indices
n, m, n̄, m̄ correspond to the indices of the Γ's in the expectation value in the given order.

The source time `t₀` is used to circularly shift `C_tnmn̄m̄` such that it is at the origin.
The four momentum indices in `Iₚ` are used for the momentum projection in the positions
x₁, x₂, x₃ and x₄.
"""
function DD_nonlocal_contractons!(
    C_tnmn̄m̄::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    Φ_kltiₚ::AbstractArray, Γ_arr::AbstractVector{<:AbstractMatrix},
    t₀::Integer, Iₚ::AbstractVector{<:Integer}; swap_ud::Bool=false
)
    # Copy array to not modify it outside of function
    Iₚ = copy(Iₚ)

    if swap_ud
        # Swap momenta at source
        permute!(Iₚ, [1, 2, 4, 3])
    end

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

    # Mode doublet at source time `t₀` for momenta `Iₚ[3]` and `Iₚ[4]`
    Φ_kl_t₀p₃ = @view Φ_kltiₚ[:, :, i_t₀, Iₚ[3]]
    Φ_kl_t₀p₄ = @view Φ_kltiₚ[:, :, i_t₀, Iₚ[4]]

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Time index for storing correlator entry
        i_Δt = mod1(iₜ-t₀, parms.Nₜ)

        # Perambulators at sink time t
        τ_charm_αkβl_t = @view τ_charm_αkβlt[:, :, :, :, iₜ]
        τ_light_αkβl_t = @view τ_light_αkβlt[:, :, :, :, iₜ]

        # Light perambulator in backward direction
        TO.@tensoropt (l, k) begin
            τ_bw_light_kαβl_t[k, α, β, l] :=
                γ[5][β, β'] * conj(τ_light_αkβl_t)[β', l, α', k] * γ[5][α', α]
        end

        # Mode doublets at sink time t for momenta `Iₚ[1]` and `Iₚ[2]`
        Φ_kl_tp₁ = @view Φ_kltiₚ[:, :, iₜ, Iₚ[1]]
        Φ_kl_tp₂ = @view Φ_kltiₚ[:, :, iₜ, Iₚ[2]]

        # Tensor contractions
        #####################

        # Disconnected parts
        TO.@tensoropt (k, k', l, l') begin
            C_disc1_nn̄[n, n̄] :=
                Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₃)[l, l'] * Γbar_αβn[β, β', n̄] *
                τ_bw_light_kαβl_t[l, β', α, k]

            C_disc2_mm̄[m, m̄] :=
                Φ_kl_tp₂[k, k'] * Γ_αβn[α, α', m] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₄)[l, l'] * Γbar_αβn[β, β', m̄] *
                τ_bw_light_kαβl_t[l, β', α, k]
        end

        # Connected part
        TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            C_conn_nmn̄m̄[n, m, n̄, m̄] := 
                Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₄)[l, l'] * Γbar_αβn[β, β', m̄] *
                τ_bw_light_kαβl_t[l, β', α_, k̃] *
                Φ_kl_tp₂[k̃, k̃'] * Γ_αβn[α_, α_', m] *
                τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                conj(Φ_kl_t₀p₃)[l̃, l̃'] * Γbar_αβn[β_, β_', n̄] *
                τ_bw_light_kαβl_t[l̃, β_', α, k]
        end

        # Combine connected and disconnected part and store it in correlator
        # (use Δt=t-t₀ as time)
        C_nmn̄m̄_Δt = @view C_tnmn̄m̄[i_Δt, :, :, :, :, ]
        if swap_ud
            # Swap Gamma matrix indices at source
            TO.@tensoropt begin
                C_nmn̄m̄_Δt[n, m, m̄, n̄] =
                    C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
            end
        else
            TO.@tensoropt begin
                C_nmn̄m̄_Δt[n, m, n̄, m̄] =
                    C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
            end
        end
    end

    return
end

@doc raw"""
    DD_nonlocal_contractons(τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray,
    Φ_kliₚ_t::AbstractArray, Φ_kliₚ_t₀::AbstractArray,
    Γ_arr::AbstractVector{<:AbstractMatrix}, Iₚ::AbstractVector{<:Integer};
    swap_ud::Bool=false) -> C_nmn̄m̄::AbstractArray

Contract the charm perambulator `τ_charm_αkβl_t`, the light perambulator `τ_light_αkβl_t`
and the mode doublets `Φ_kliₚ_t` and `Φ_kliₚ_t₀` to get the nolocal DD correlator and return
it. These arrays are assumed to only contain data for a single sink time `t` and source time
`t₀`. The matrices in `Γ_arr` are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum
expectation value of the form (in position space) \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃u)(x₃) (c̄Γbar₄d)(x₄)>` \
 or \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃d)(x₃) (c̄Γbar₄u)(x₄)>` \
if `swap_ud` = true.
    
The result is returned as the array `C_nmn̄m̄` where the indices n, m, n̄, m̄ correspond to
the indices of the Γ's in the expectation value in the given order.

The four momentum indices in `Iₚ` are used for the momentum projection in the positions
x₁, x₂, x₃ and x₄.
"""
function DD_nonlocal_contractons(
    τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray, Φ_kliₚ_t::AbstractArray,
    Φ_kliₚ_t₀::AbstractArray, Γ_arr::AbstractVector{<:AbstractMatrix},
    Iₚ::AbstractVector{<:Integer}; swap_ud::Bool=false
)
    # Copy array to not modify it outside of function
    Iₚ = copy(Iₚ)

    if swap_ud
        # Swap momenta at source
        permute!(Iₚ, [1, 2, 4, 3])
    end

    # Number of gamma matrices
    Nᵧ = length(Γ_arr)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

    # Allocate correlator
    C_nmn̄m̄ = Array{ComplexF64}(undef, Nᵧ, Nᵧ, Nᵧ, Nᵧ)

    # Mode doublet at source for momenta `Iₚ[3]` and `Iₚ[4]`
    Φ_kl_t₀p₃ = @view Φ_kliₚ_t₀[:, :, Iₚ[3]]
    Φ_kl_t₀p₄ = @view Φ_kliₚ_t₀[:, :, Iₚ[4]]

    # Light perambulator in backward direction
    TO.@tensoropt (l, k) begin
        τ_bw_light_kαβl_t[k, α, β, l] :=
            γ[5][β, β'] * conj(τ_light_αkβl_t)[β', l, α', k] * γ[5][α', α]
    end

    # Mode doublets at sink for momenta `Iₚ[1]` and `Iₚ[2]`
    Φ_kl_tp₁ = @view Φ_kliₚ_t[:, :, Iₚ[1]]
    Φ_kl_tp₂ = @view Φ_kliₚ_t[:, :, Iₚ[2]]

    # Tensor contractions
    #####################

    # Disconnected parts
    TO.@tensoropt (k, k', l, l') begin
        C_disc1_nn̄[n, n̄] :=
            Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
            τ_charm_αkβl_t[α', k', β, l'] *
            conj(Φ_kl_t₀p₃)[l, l'] * Γbar_αβn[β, β', n̄] *
            τ_bw_light_kαβl_t[l, β', α, k]

        C_disc2_mm̄[m, m̄] :=
            Φ_kl_tp₂[k, k'] * Γ_αβn[α, α', m] *
            τ_charm_αkβl_t[α', k', β, l'] *
            conj(Φ_kl_t₀p₄)[l, l'] * Γbar_αβn[β, β', m̄] *
            τ_bw_light_kαβl_t[l, β', α, k]
    end

    # Connected part
    TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
        C_conn_nmn̄m̄[n, m, n̄, m̄] := 
            Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
            τ_charm_αkβl_t[α', k', β, l'] *
            conj(Φ_kl_t₀p₄)[l, l'] * Γbar_αβn[β, β', m̄] *
            τ_bw_light_kαβl_t[l, β', α_, k̃] *
            Φ_kl_tp₂[k̃, k̃'] * Γ_αβn[α_, α_', m] *
            τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
            conj(Φ_kl_t₀p₃)[l̃, l̃'] * Γbar_αβn[β_, β_', n̄] *
            τ_bw_light_kαβl_t[l̃, β_', α, k]
    end

    # Combine connected and disconnected part and store it in correlator
    # (use Δt=t-t₀ as time)
    if swap_ud
        # Swap Gamma matrix indices at source
        TO.@tensoropt begin
            C_nmn̄m̄[n, m, m̄, n̄] =
                C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
        end
    else
        TO.@tensoropt begin
            C_nmn̄m̄[n, m, n̄, m̄] =
                C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
        end
    end

    return C_nmn̄m̄
end

@doc raw"""
    DD_mixed_contractons!(Cₙₗ_tnmn̄m̄iₚ::AbstractArray, Cₗₙ_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, Iₚ_nonlocal::AbstractVector{<:Integer}, p_local_arr::AbstractVector{<:AbstractVector})

Contract the charm perambulator `τ_charm_αkβlt`, the light perambulator `τ_light_αkβlt`,
the mode doublets `Φ_kltiₚ` and the sparse Laplace modes in `sparse_modes_arrays` to
get the mixed DD correlators. The matrices in `Γ_arr` are the matrices in the interpolating
operators. The correlator is computed for all possible combinations of them. This gives
vacuum expectation values of the form (in position space) \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃u c̄Γbar₄d)(x)>` \
(nonlocal-local) and \
`<(ūΓ₁c d̄Γ₂c)(x) (c̄Γbar₃u)(x₁) (c̄Γbar₄d)(x₂)>` \
(local-nonlocal). The first one is stored in `Cₙₗ_tnmn̄m̄iₚ` and the second one in
`Cₗₙ_tnmn̄m̄iₚ` where the indices n, m, n̄, m̄ correspond to the indices of the Γ's in the
expectation values in the given order.

The source time `t₀` is used to circularly shift the correlators such that it is at the
origin. The two momentum indices in `Iₚ_nonlocal` are used for the momentum projection in
the nonlocal operator (positions x₁ and x₂). The array `p_local_arr` contains the integer
momenta for the momentum projection in the local operator (index iₚ in `Cₙₗ_tnmn̄m̄iₚ` and
`Cₗₙ_tnmn̄m̄iₚ`).
"""
function DD_mixed_contractons!(
    Cₙₗ_tnmn̄m̄iₚ::AbstractArray, Cₗₙ_tnmn̄m̄iₚ::AbstractArray,
    τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray,
    Φ_kltiₚ::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray},
    Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer,
    Iₚ_nonlocal::AbstractVector{<:Integer}, p_local_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Number of points on spares lattice
    _, N_points, _ = size(x_sink_μiₓt)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

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
        # Time index for storing correlator entry
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

        # Contract mode doublet and charm perambulator
        TO.@tensoropt (k, k', l, l') begin
            # For nonlocal-local correlator
            Φτ_charm_kαβl_tp₁[k, α, β, l] := Φ_kl_tp₁[k, k'] * τ_charm_αkβl_t[α, k', β, l]
            Φτ_charm_kαβl_tp₂[k, α, β, l] := Φ_kl_tp₂[k, k'] * τ_charm_αkβl_t[α, k', β, l]

            # for local-nonlocal correlator
            τΦ_charm_kαβl_t₀p₁[k, α, β, l] :=
                τ_charm_αkβl_t[α, k, β, l'] * conj(Φ_kl_t₀p₁)[l, l']
            τΦ_charm_kαβl_t₀p₂[k, α, β, l] :=
                τ_charm_αkβl_t[α, k, β, l'] * conj(Φ_kl_t₀p₂)[l, l']
        end

        # Nonlocal-local tensor contractions
        ####################################

        # Loop over source position iₓ
        for iₓ in 1:N_points
            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Pre-contractions
            TO.@tensoropt (k, l) begin
                Φτv_charm_kαβc_iₓtp₁[k, α, β, a] := 
                    Φτ_charm_kαβl_tp₁[k, α, β, l] * conj(v_src_ck_iₓt₀)[a, l]
                Φτv_charm_kαβc_iₓtp₂[k, α, β, a] := 
                    Φτ_charm_kαβl_tp₂[k, α, β, l] * conj(v_src_ck_iₓt₀)[a, l]
                vτ_bw_light_αcβk_iₓt[β, a, α, k] := 
                    v_src_ck_iₓt₀[a, l] * τ_bw_light_kαβl_t[l, β, α, k]
            end

            # Disconnected part
            #= TO.@tensoropt (l, k, l', k') begin
                C_disc1_nn̄[n, n̄] :=
                    Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                    τ_charm_αkβl_t[α', k', β, l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                    Γbar_αβn[β, β', n̄] * τ_bw_light_kαβl_t[l, β', α, k]
                
                C_disc2_mm̄[m, m̄] :=
                    Φ_kl_tp₂[k, k'] * Γ_αβn[α, α', m] *
                    τ_charm_αkβl_t[α', k', β, l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                    Γbar_αβn[β, β', m̄] * τ_bw_light_kαβl_t[l, β', α, k]
            end =#
            TO.@tensoropt (l, k, l', k') begin
                C_disc1_nn̄[n, n̄] :=
                    Γ_αβn[α, α', n] * Φτv_charm_kαβc_iₓtp₁[k, α', β, b] *
                    Γbar_αβn[β, β', n̄] * vτ_bw_light_αcβk_iₓt[β', b, α, k]
                
                C_disc2_mm̄[m, m̄] :=
                    Γ_αβn[α, α', m] * Φτv_charm_kαβc_iₓtp₂[k, α', β, a] *
                    Γbar_αβn[β, β', m̄] * vτ_bw_light_αcβk_iₓt[β', a, α, k]
            end

            # Connected part
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                    τ_charm_αkβl_t[α', k', β, l'] *
                    conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                    Γbar_αβn[β, β', m̄] * τ_bw_light_kαβl_t[l, β', α_, k̃] *
                    Φ_kl_tp₂[k̃, k̃'] * Γ_αβn[α_, α_', m] *
                    τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                    conj(v_src_ck_iₓt₀)[b̃, l̃'] * v_src_ck_iₓt₀[b̃, l̃] *
                    Γbar_αβn[β_, β_', n̄] * τ_bw_light_kαβl_t[l̃, β_', α, k]
            end =#
            TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    Γ_αβn[α, α', n] * Φτv_charm_kαβc_iₓtp₁[k, α', β, b] *
                    Γbar_αβn[β, β', m̄] * vτ_bw_light_αcβk_iₓt[β', b, α_, k̃] *
                    Γ_αβn[α_, α_', m] * Φτv_charm_kαβc_iₓtp₂[k̃, α_', β_, b̃] *
                    Γbar_αβn[β_, β_', n̄] * vτ_bw_light_αcβk_iₓt[β_', b̃, α, k]
            end

            # Combine connected and disconnected part
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
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

        # Loop over sink position iₓ′
        for iₓ′ in 1:N_points
            # Laplace modes at sink time t and position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

            # Pre-contractions
            TO.@tensoropt (k, l) begin
                vτΦ_charm_cαβl_iₓ′t₀p₁[a, α, β, l] :=
                    v_sink_ck_iₓ′t[a, k] * τΦ_charm_kαβl_t₀p₁[k, α, β, l]
                vτΦ_charm_cαβl_iₓ′t₀p₂[a, α, β, l] :=
                    v_sink_ck_iₓ′t[a, k] * τΦ_charm_kαβl_t₀p₂[k, α, β, l]
                τv_bw_light_kαβc[l, α, β, a] :=
                    τ_bw_light_kαβl_t[l, α, β, k] * conj(v_sink_ck_iₓ′t)[a, k]
            end

            # Disconnected part
            #= TO.@tensoropt (l, k, l', k') begin
                C_disc1_nn̄[n, n̄] :=
                    conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                    Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                    conj(Φ_kl_t₀p₁)[l, l'] * Γbar_αβn[β, β', n̄] *
                    τ_bw_light_kαβl_t[l, β', α, k]
                
                C_disc2_mm̄[m, m̄] :=
                    conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                    Γ_αβn[α, α', m] * τ_charm_αkβl_t[α', k', β, l'] *
                    conj(Φ_kl_t₀p₂)[l, l'] * Γbar_αβn[β, β', m̄] *
                    τ_bw_light_kαβl_t[l, β', α, k]
            end =#
            TO.@tensoropt (l, k, l', k') begin
                C_disc1_nn̄[n, n̄] :=
                    Γ_αβn[α, α', n] * vτΦ_charm_cαβl_iₓ′t₀p₁[a, α', β, l] * 
                    Γbar_αβn[β, β', n̄] *τv_bw_light_kαβc[l, β', α, a]
                
                C_disc2_mm̄[m, m̄] :=
                    Γ_αβn[α, α', m] * vτΦ_charm_cαβl_iₓ′t₀p₂[a, α', β, l] * 
                    Γbar_αβn[β, β', m̄] * τv_bw_light_kαβc[l, β', α, a]
            end

            # Connected part
            #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                    Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                    conj(Φ_kl_t₀p₂)[l, l'] * Γbar_αβn[β, β', m̄] *
                    τ_bw_light_kαβl_t[l, β', α_, k̃] *
                    conj(v_sink_ck_iₓ′t)[ã, k̃] * v_sink_ck_iₓ′t[ã, k̃'] *
                    Γ_αβn[α_, α_', m] * τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                    conj(Φ_kl_t₀p₁)[l̃, l̃'] * Γbar_αβn[β_, β_', n̄] *
                    τ_bw_light_kαβl_t[l̃, β_', α, k]
            end =#
            TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
                C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                    Γ_αβn[α, α', n] * vτΦ_charm_cαβl_iₓ′t₀p₂[a, α', β, l] * 
                    Γbar_αβn[β, β', m̄] * τv_bw_light_kαβc[l, β', α_, ã] * 
                    Γ_αβn[α_, α_', m] * vτΦ_charm_cαβl_iₓ′t₀p₁[ã, α_', β_, l̃] * 
                    Γbar_αβn[β_, β_', n̄] * τv_bw_light_kαβc[l̃, β_', α, a]
            end

            # Combine connected and disconnected part
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
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
    Cₙₗ_tnmn̄m̄iₚ .*= prod(parms.Nₖ)/N_points
    Cₗₙ_tnmn̄m̄iₚ .*= prod(parms.Nₖ)/N_points

    return
end

@doc raw"""
    DD_mixed_contractons(τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray, Φ_kliₚ_t::AbstractArray,
    Φ_kliₚ_t₀::AbstractArray, sparse_modes_arrays_tt₀::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, Iₚ_nonlocal::AbstractVector{<:Integer}, p_local_arr::AbstractVector{<:AbstractVector}) -> (Cₙₗ_nmn̄m̄iₚ::AbstractArray, Cₗₙ_nmn̄m̄iₚ::AbstractArray)

Contract the charm perambulator `τ_charm_αkβl_t`, the light perambulator `τ_light_αkβl_t`,
the mode doublets `Φ_kliₚ_t` and `Φ_kliₚ_t₀` and the sparse Laplace modes in
`sparse_modes_arrays_tt₀` to get the mixed DD correlators and return them.  These arrays are
assumed to only contain data for a single sink time `t` and source time `t₀`.
The matrices in `Γ_arr` are the matrices in the interpolating operators. The correlator is
computed for all possible combinations of them. This gives vacuum expectation values of the
form (in position space) \
`<(ūΓ₁c)(x₁) (d̄Γ₂c)(x₂) (c̄Γbar₃u c̄Γbar₄d)(x)>` \
(nonlocal-local) and \
`<(ūΓ₁c d̄Γ₂c)(x) (c̄Γbar₃u)(x₁) (c̄Γbar₄d)(x₂)>` \
(local-nonlocal). The first one is returned as the array `Cₙₗ_nmn̄m̄iₚ` and the second one as
the array `Cₗₙ_nmn̄m̄iₚ` where the indices n, m, n̄, m̄ correspond to the indices of the Γ's in the
expectation values in the given order.

The two momentum indices in `Iₚ_nonlocal` are used for the momentum projection in
the nonlocal operator (positions x₁ and x₂). The array `p_local_arr` contains the integer
momenta for the momentum projection in the local operator (index iₚ in `Cₙₗ_nmn̄m̄iₚ` and
`Cₗₙ_nmn̄m̄iₚ`).
"""
function DD_mixed_contractons(
    τ_charm_αkβl_t::AbstractArray, τ_light_αkβl_t::AbstractArray, Φ_kliₚ_t::AbstractArray,
    Φ_kliₚ_t₀::AbstractArray, sparse_modes_arrays_tt₀::NTuple{4, AbstractArray},
    Γ_arr::AbstractVector{<:AbstractMatrix}, Iₚ_nonlocal::AbstractVector{<:Integer},
    p_local_arr::AbstractVector{<:AbstractVector}
)
    # Unpack sparse modes arrays
    x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀ = sparse_modes_arrays_tt₀

    # Number of points on spares lattice
    N_points = size(x_sink_μiₓ_t, 2)

    # Number of gamma matrices
    Nᵧ = length(Γ_arr)

    # Convert vector of γ-matrices to contiguous array and compute Γbar matrices
    Γ_αβn = stack(Γ_arr)
    TO.@tensoropt Γbar_αβn[α, β, n] := γ[4][α, α'] * conj(Γ_αβn)[β', α', n] * γ[4][β', β]

    # Convert momentum array to contiguous array
    p_local_μiₚ = stack(p_local_arr)

    # Allocate correlator
    Cₙₗ_nmn̄m̄iₚ = zeros(ComplexF64, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(p_local_arr))
    Cₗₙ_nmn̄m̄iₚ = zeros(ComplexF64, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(p_local_arr))

    # Mode doublet at source time `t₀` for momenta `Iₚ_nonlocal[1]` and `Iₚ_nonlocal[2]`
    Φ_kl_t₀p₁ = @view Φ_kliₚ_t₀[:, :, Iₚ_nonlocal[1]]
    Φ_kl_t₀p₂ = @view Φ_kliₚ_t₀[:, :, Iₚ_nonlocal[2]]

    # Mode doublets at sink time t for momenta `Iₚ_nonlocal[1]` and `Iₚ_nonlocal[2]`
    Φ_kl_tp₁ = @view Φ_kliₚ_t[:, :, Iₚ_nonlocal[1]]
    Φ_kl_tp₂ = @view Φ_kliₚ_t[:, :, Iₚ_nonlocal[2]]

    # Light perambulator in backward direction
    TO.@tensoropt (l, k) begin
        τ_bw_light_kαβl_t[k, α, β, l] :=
            γ[5][β, β'] * conj(τ_light_αkβl_t)[β', l, α', k] * γ[5][α', α]
    end

    # Contract mode doublet and charm perambulator
    TO.@tensoropt (k, k', l, l') begin
        # For nonlocal-local correlator
        Φτ_charm_kαβl_tp₁[k, α, β, l] := Φ_kl_tp₁[k, k'] * τ_charm_αkβl_t[α, k', β, l]
        Φτ_charm_kαβl_tp₂[k, α, β, l] := Φ_kl_tp₂[k, k'] * τ_charm_αkβl_t[α, k', β, l]

        # for local-nonlocal correlator
        τΦ_charm_kαβl_t₀p₁[k, α, β, l] :=
            τ_charm_αkβl_t[α, k, β, l'] * conj(Φ_kl_t₀p₁)[l, l']
        τΦ_charm_kαβl_t₀p₂[k, α, β, l] :=
            τ_charm_αkβl_t[α, k, β, l'] * conj(Φ_kl_t₀p₂)[l, l']
    end

    # Nonlocal-local tensor contractions
    ####################################

    # Loop over source position iₓ
    for iₓ in 1:N_points
        # Laplace modes at src time t₀ and position iₓ
        v_src_ck_iₓt₀ = @view v_src_ciₓk_t₀[:, iₓ, :]

        # Pre-contractions
        TO.@tensoropt (k, l) begin
            Φτv_charm_kαβc_iₓtp₁[k, α, β, a] := 
                Φτ_charm_kαβl_tp₁[k, α, β, l] * conj(v_src_ck_iₓt₀)[a, l]
            Φτv_charm_kαβc_iₓtp₂[k, α, β, a] := 
                Φτ_charm_kαβl_tp₂[k, α, β, l] * conj(v_src_ck_iₓt₀)[a, l]
            vτ_bw_light_αcβk_iₓt[β, a, α, k] := 
                v_src_ck_iₓt₀[a, l] * τ_bw_light_kαβl_t[l, β, α, k]
        end

        # Disconnected part
        #= TO.@tensoropt (l, k, l', k') begin
            C_disc1_nn̄[n, n̄] :=
                Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                Γbar_αβn[β, β', n̄] * τ_bw_light_kαβl_t[l, β', α, k]
            
            C_disc2_mm̄[m, m̄] :=
                Φ_kl_tp₂[k, k'] * Γ_αβn[α, α', m] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                Γbar_αβn[β, β', m̄] * τ_bw_light_kαβl_t[l, β', α, k]
        end =#
        TO.@tensoropt (l, k, l', k') begin
            C_disc1_nn̄[n, n̄] :=
                Γ_αβn[α, α', n] * Φτv_charm_kαβc_iₓtp₁[k, α', β, b] *
                Γbar_αβn[β, β', n̄] * vτ_bw_light_αcβk_iₓt[β', b, α, k]
            
            C_disc2_mm̄[m, m̄] :=
                Γ_αβn[α, α', m] * Φτv_charm_kαβc_iₓtp₂[k, α', β, a] *
                Γbar_αβn[β, β', m̄] * vτ_bw_light_αcβk_iₓt[β', a, α, k]
        end

        # Connected part
        #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                Φ_kl_tp₁[k, k'] * Γ_αβn[α, α', n] *
                τ_charm_αkβl_t[α', k', β, l'] *
                conj(v_src_ck_iₓt₀)[b, l'] * v_src_ck_iₓt₀[b, l] *
                Γbar_αβn[β, β', m̄] * τ_bw_light_kαβl_t[l, β', α_, k̃] *
                Φ_kl_tp₂[k̃, k̃'] * Γ_αβn[α_, α_', m] *
                τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                conj(v_src_ck_iₓt₀)[b̃, l̃'] * v_src_ck_iₓt₀[b̃, l̃] *
                Γbar_αβn[β_, β_', n̄] * τ_bw_light_kαβl_t[l̃, β_', α, k]
        end =#
        TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                Γ_αβn[α, α', n] * Φτv_charm_kαβc_iₓtp₁[k, α', β, b] *
                Γbar_αβn[β, β', m̄] * vτ_bw_light_αcβk_iₓt[β', b, α_, k̃] *
                Γ_αβn[α_, α_', m] * Φτv_charm_kαβc_iₓtp₂[k̃, α_', β_, b̃] *
                Γbar_αβn[β_, β_', n̄] * vτ_bw_light_αcβk_iₓt[β_', b̃, α, k]
        end

        # Combine connected and disconnected part
        TO.@tensoropt begin
            C_nmn̄m̄[n, m, n̄, m̄] :=
                C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
        end

        # Momentum projection
        p2πix = 2π*im * 
            x_src_μiₓ_t₀[:, iₓ]./parms.Nₖ
        exp_ipx_arr = exp.(p_local_μiₚ' * p2πix)
        for iₚ in eachindex(p_local_arr)
            Cₙₗ_nmn̄m̄_iₚ = @view Cₙₗ_nmn̄m̄iₚ[:, :, :, :, iₚ]

            exp_ipx = exp_ipx_arr[iₚ]
            TO.@tensoropt begin
                Cₙₗ_nmn̄m̄_iₚ[n, m, n̄, m̄] += exp_ipx * C_nmn̄m̄[n, m, n̄, m̄]
            end
        end
    end

    # Local-nonlocal tensor contractions
    ####################################

    # Loop over sink position iₓ′
    for iₓ′ in 1:N_points
        # Laplace modes at sink time t and position iₓ′
        v_sink_ck_iₓ′t = @view v_sink_ciₓk_t[:, iₓ′, :]

        # Pre-contractions
        TO.@tensoropt (k, l) begin
            vτΦ_charm_cαβl_iₓ′t₀p₁[a, α, β, l] :=
                v_sink_ck_iₓ′t[a, k] * τΦ_charm_kαβl_t₀p₁[k, α, β, l]
            vτΦ_charm_cαβl_iₓ′t₀p₂[a, α, β, l] :=
                v_sink_ck_iₓ′t[a, k] * τΦ_charm_kαβl_t₀p₂[k, α, β, l]
            τv_bw_light_kαβc[l, α, β, a] :=
                τ_bw_light_kαβl_t[l, α, β, k] * conj(v_sink_ck_iₓ′t)[a, k]
        end

        # Disconnected part
        #= TO.@tensoropt (l, k, l', k') begin
            C_disc1_nn̄[n, n̄] :=
                conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₁)[l, l'] * Γbar_αβn[β, β', n̄] *
                τ_bw_light_kαβl_t[l, β', α, k]
            
            C_disc2_mm̄[m, m̄] :=
                conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                Γ_αβn[α, α', m] * τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₂)[l, l'] * Γbar_αβn[β, β', m̄] *
                τ_bw_light_kαβl_t[l, β', α, k]
        end =#
        TO.@tensoropt (l, k, l', k') begin
            C_disc1_nn̄[n, n̄] :=
                Γ_αβn[α, α', n] * vτΦ_charm_cαβl_iₓ′t₀p₁[a, α', β, l] * 
                Γbar_αβn[β, β', n̄] *τv_bw_light_kαβc[l, β', α, a]
            
            C_disc2_mm̄[m, m̄] :=
                Γ_αβn[α, α', m] * vτΦ_charm_cαβl_iₓ′t₀p₂[a, α', β, l] * 
                Γbar_αβn[β, β', m̄] * τv_bw_light_kαβc[l, β', α, a]
        end

        # Connected part
        #= TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                conj(v_sink_ck_iₓ′t)[a, k] * v_sink_ck_iₓ′t[a, k'] *
                Γ_αβn[α, α', n] * τ_charm_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀p₂)[l, l'] * Γbar_αβn[β, β', m̄] *
                τ_bw_light_kαβl_t[l, β', α_, k̃] *
                conj(v_sink_ck_iₓ′t)[ã, k̃] * v_sink_ck_iₓ′t[ã, k̃'] *
                Γ_αβn[α_, α_', m] * τ_charm_αkβl_t[α_', k̃', β_, l̃'] *
                conj(Φ_kl_t₀p₁)[l̃, l̃'] * Γbar_αβn[β_, β_', n̄] *
                τ_bw_light_kαβl_t[l̃, β_', α, k]
        end =#
        TO.@tensoropt (l, k, l', k', l̃, k̃, l̃', k̃') begin
            C_conn_nmn̄m̄[n, m, n̄, m̄] :=
                Γ_αβn[α, α', n] * vτΦ_charm_cαβl_iₓ′t₀p₂[a, α', β, l] * 
                Γbar_αβn[β, β', m̄] * τv_bw_light_kαβc[l, β', α_, ã] * 
                Γ_αβn[α_, α_', m] * vτΦ_charm_cαβl_iₓ′t₀p₁[ã, α_', β_, l̃] * 
                Γbar_αβn[β_, β_', n̄] * τv_bw_light_kαβc[l̃, β_', α, a]
        end

        # Combine connected and disconnected part
        TO.@tensoropt begin
            C_nmn̄m̄[n, m, n̄, m̄] :=
                C_disc1_nn̄[n, n̄]*C_disc2_mm̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
        end

        # Momentum projection
        m2πix = -2π*im * 
            x_sink_μiₓ_t[:, iₓ′]./parms.Nₖ
        exp_ipx_arr = exp.(p_local_μiₚ' * m2πix)
        for iₚ in eachindex(p_local_arr)
            Cₗₙ_nmn̄m̄_iₚ = @view Cₗₙ_nmn̄m̄iₚ[:, :, :, :, iₚ]

            exp_ipx = exp_ipx_arr[iₚ]
            TO.@tensoropt begin
                Cₗₙ_nmn̄m̄_iₚ[n, m, n̄, m̄] += exp_ipx * C_nmn̄m̄[n, m, n̄, m̄]
            end
        end
    end

    # Normalization
    Cₙₗ_nmn̄m̄iₚ .*= prod(parms.Nₖ)/N_points
    Cₗₙ_nmn̄m̄iₚ .*= prod(parms.Nₖ)/N_points

    return Cₙₗ_nmn̄m̄iₚ, Cₗₙ_nmn̄m̄iₚ
end
