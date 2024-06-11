########################
# Locale DD Contractions
########################

@doc raw"""
    DD_local_contractons!(C_tnmn̄m̄iₚ::AbstractArray, τ_charm_αkβlt::AbstractArray, τ_light_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, Γ_arr::AbstractVector{<:AbstractMatrix}, t₀::Integer, p_arr::AbstractVector{<:AbstractVector}

Contract the charm perambulator `τ_charm_αkβlt` and the light perambulator `τ_light_αkβlt`
and the sparse Laplace modes stored in `sparse_modes_arrays` to get the local DD
correlator. The matrices in `Γ_arr` are the matrices in the interpolating operators.
The correlator is computed for all possible combinations of them. This gives a vacuum expectation value of the form \
`<(ūΓ₁c d̄Γ₂c)(x) (c̄Γbar₃u c̄Γbar₄d)(0)>` \
(in position space). The result is stored in `C_tnmn̄m̄iₚ` where the indices n, m, n̄, m̄
correspond to the indices of the Γ's in the expectation value in the given order.

The source time `t₀` is used to circularly shift `C_tnmn̄m̄iₚ` such that the source time is at
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

    # Set correlator C_tnmn̄m̄iₚ to zero
    C_tnmn̄m̄iₚ .= 0

    # Loop over all sink time indices (using multithreading)
    Threads.@threads for iₜ in 1:parms.Nₜ
        # Perambulators, sink position and Laplace modes at sink time t (index `iₜ`)
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
            # Laplace modes at sink time t (index `iₜ`) and position iₓ′
            v_sink_ck_iₓ′t = @view v_sink_ciₓkt[:, iₓ′, :, iₜ]

            # Laplace modes at src time t₀ and position iₓ
            v_src_ck_iₓt₀ = @view v_src_ciₓkt[:, iₓ, :, i_t₀]

            # Tensor contractions
            #####################

            # Smeared charm propagator
            TO.@tensoropt (k, l) begin
                D⁻¹_charm_αaβb[α, a, β, b] :=
                    v_sink_ck_iₓ′t[a, k] * 
                    τ_charm_αkβl_t[α, k, β, l] * conj(v_src_ck_iₓt₀)[b, l]
            end

            # Smeared light propagator (adjoint)
            TO.@tensoropt (k, l) begin
                D⁻¹′_light_αaβb[α, a, β, b] :=
                    v_src_ck_iₓt₀[a, k] *
                    γ₅τ_conjγ₅_light_αkβl_t[β, l, α, k] * conj(v_sink_ck_iₓ′t)[b, l]
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
                    Γ_αβn[α, α', n] * D⁻¹_charm_αaβb[α', a, β, b] *
                    Γbar_αβn[β, β', n̄] * D⁻¹′_light_αaβb[β', b, α, a]
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
                    Γ_αβn[α, α', n] * D⁻¹_charm_αaβb[α', a, β, b] *
                    Γbar_αβn[β, β', m̄] * D⁻¹′_light_αaβb[β', b, α_, ã] *
                    Γ_αβn[α_, α_', m] * D⁻¹_charm_αaβb[α_', ã, β_, b̃] *
                    Γbar_αβn[β_, β_', n̄] * D⁻¹′_light_αaβb[β_', b̃, α, a]
            end

            # Combine connected and disconnected part
            TO.@tensoropt begin
                C_nmn̄m̄[n, m, n̄, m̄] :=
                    C_disc_nn̄[n, n̄]*C_disc_nn̄[m, m̄] - C_conn_nmn̄m̄[n, m, n̄, m̄]
            end

            # Momentum projection
            m2πiΔx = -2π*im * 
                (x_sink_μiₓt[:, iₓ′, iₜ] - x_src_μiₓt[:, iₓ, i_t₀])./parms.Nₖ
            i_Δt = mod1(iₜ-t₀, parms.Nₜ)
            exp_mipΔx_arr = exp.(p_μiₚ' * m2πiΔx)
            for (iₚ, exp_mipΔx) in enumerate(exp_mipΔx_arr)
                # Use Δt as time such that t₀=0
                C_nmn̄m̄_Δtiₚ = @view C_tnmn̄m̄iₚ[i_Δt, :, :, :, :, iₚ]
                TO.@tensoropt begin
                    C_nmn̄m̄_Δtiₚ[n, m, n̄, m̄] += exp_mipΔx * C_nmn̄m̄[n, m, n̄, m̄]
                end
            end
        end
        end
    end

    # Normalization
    C_tnmn̄m̄iₚ *= (prod(parms.Nₖ)/N_points)^2

    return
end

