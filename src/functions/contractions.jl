###########################
# Pseudoscalar Contractions
###########################

@doc raw"""
    pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, t₀::Integer)
    pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, t₀::Integer)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` to get the pseudoscalar correlator
for zero momentum where `τ₁_αkβlt` is used to propagate in forward and `τ₂_αkβlt` in
backward direction. This gives a vacuum expectation value of the form \
`<(ψ₂γ₅ψ₁)(x) (ψ₁γ₅ψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

If only one perambulator `τ_αkβlt` is given, it is used to propagete in
both directions.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin.
"""
function pseudoscalar_contraction_p0!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, t₀::Integer
)
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # perambulators at sink (index `iₜ`)
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :, iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = τ₁_αkβl_t[α, k, β, l] * conj(τ₂_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end
pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, t₀::Integer) =
    pseudoscalar_contraction_p0!(Cₜ, τ_αkβlt, τ_αkβlt, t₀)

@doc raw"""
    pseudoscalar_contraction!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray, t₀::Integer, iₚ::Integer)
    pseudoscalar_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray, t₀::Integer, iₚ::Integer)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` and the mode doublets `Φ_kltiₚ` to get
the pseudoscalar correlator where `τ₁_αkβlt` is used to propagate in forward and `τ₂_αkβlt`
in backward direction. This gives a vacuum expectation value of the form \
`<(ψ₂γ₅ψ₁)(x) (ψ₁γ₅ψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

If only one perambulator `τ_αkβlt` is given, it is used to propagete
in both directions.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin. The index `iₚ` sets the momentum that is used from
the mode doublets.
"""
function pseudoscalar_contraction!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray,
    Φ_kltiₚ::AbstractArray, t₀::Integer, iₚ::Integer
)
    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Mode doublet at source time `t₀`
    Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Mode doublet and perambulators at sink time t (index `iₜ`)
        Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :,iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :,iₜ]
        
        # Tensor contraction
        TO.@tensoropt begin
            C = Φ_kl_tiₚ[k, k'] * τ₁_αkβl_t[α, k', β, l'] *
                conj(Φ_kl_t₀iₚ[l, l']) * conj(τ₂_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end
pseudoscalar_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                          Φ_kltiₚ::AbstractArray, t₀::Integer, iₚ::Integer) = 
    pseudoscalar_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)

@doc raw"""
    pseudoscalar_sparse_contraction!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, t₀::Integer, p::Vector)
    pseudoscalar_sparse_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray}, t₀::Integer, p::AbstractVector)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` and the sparse Laplace modes stored in
`sparse_modes_arrays` to get the pseudoscalar correlator where `τ₁_αkβlt` is used to
propagate in forward and `τ₂_αkβlt` in backward direction. This gives a vacuum expectation
value of the form \
`<(ψ₂γ₅ψ₁)(x) (ψ₁γ₅ψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

If only one perambulator
`τ_αkβlt` is given, it is used to propagete in both directions.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin. The array `p` is the integer momentum that is used
for the momentum projection of the correlator.
"""
function pseudoscalar_sparse_contraction!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, t₀::Integer, p::AbstractVector
)
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Source position and Laplace modes at source time `t₀`
    x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

    # Compute exp(±ipx) and reshape it to match shape of Laplace modes
    exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./parms.Nₖ)'*p)
    exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))
    exp_ipx_src_iₓ = exp.(2π*im * (x_src_μiₓ_t₀./parms.Nₖ)'*p)
    exp_ipx_src_iₓ = reshape(exp_ipx_src_iₓ, (1, N_points, 1))
    
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulators and Laplace modes at sink time t (index `iₜ`)
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :,iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :,iₜ]
        v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = conj(v_sink_ciₓk_t[a, iₓ', k]) * 
                (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ', k'] *
                τ₁_αkβl_t[α, k', β, l'] *
                conj(v_src_ciₓk_t₀[b, iₓ, l']) *
                (exp_ipx_src_iₓ .* v_src_ciₓk_t₀)[b, iₓ, l] *
                conj(τ₂_αkβl_t[α, k, β, l])
        end

        # Normalization
        C *= (prod(parms.Nₖ)/N_points)^2

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end
pseudoscalar_sparse_contraction!(
    Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, t₀::Integer, p::AbstractVector
) = pseudoscalar_sparse_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, sparse_modes_arrays, t₀, p)



################################
# Meson Contractions (connected)
################################

@doc raw"""
    meson_connected_contraction_p0!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, Γ::AbstractMatrix, Γbar::AbstractMatrix, t₀::Integer)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` to get the connected meson correlator
for zero momentum where `τ₁_αkβlt` is used to propagate in forward and `τ₂_αkβlt` in
backward direction. The matrices `Γ` and `Γbar` are the matrices in the interpolating
operators. This gives a vacuum expectation value of the form \
`<(ψ₂Γψ₁)(x) (ψ₁Γbarψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin.
"""
function meson_connected_contraction_p0!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, Γ::AbstractMatrix,
    Γbar::AbstractMatrix, t₀::Integer
)
    # Multiply γ₅ to Gammas
    γ₅Γ = γ[5]*Γ
    Γbarγ₅ = -Γbar*γ[5]

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # perambulators at sink (index `iₜ`)
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :, iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = γ₅Γ[α, α'] * τ₁_αkβl_t[α', k, β, l] *
                Γbarγ₅[β, β'] * conj(τ₂_αkβl_t[α, k, β', l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    meson_connected_contraction!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray,  Γ::AbstractMatrix, Γbar::AbstractMatrix, t₀::Integer, iₚ::Integer)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` and the mode doublets `Φ_kltiₚ` to get
the connected meson correlator where `τ₁_αkβlt` is used to propagate in forward and
`τ₂_αkβlt` in backward direction. The matrices `Γ` and `Γbar` are the matrices in the
interpolating operators. This gives a vacuum expectation value of the form \
`<(ψ₂Γψ₁)(x) (ψ₁Γbarψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin. The index `iₚ` sets the momentum that is used from
the mode doublets.
"""
function meson_connected_contraction!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray,
    Φ_kltiₚ::AbstractArray,  Γ::AbstractMatrix, Γbar::AbstractMatrix,
    t₀::Integer, iₚ::Integer
)
    # Multiply γ₅ to Gammas
    γ₅Γ = γ[5]*Γ
    Γbarγ₅ = -Γbar*γ[5]

    # Index for source time `t₀`
    i_t₀ = t₀+1

    # Mode doublet at source time `t₀`
    Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Mode doublet and perambulators at sink time t (index `iₜ`)
        Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :,iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :,iₜ]
        
        # Tensor contraction
        TO.@tensoropt begin
            C = Φ_kl_tiₚ[k, k'] *
                γ₅Γ[α, α'] * τ₁_αkβl_t[α', k', β, l'] *
                conj(Φ_kl_t₀iₚ[l, l']) * 
                Γbarγ₅[β, β'] * conj(τ₂_αkβl_t[α, k, β', l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    meson_connected_sparse_contraction!(Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray, sparse_modes_arrays::NTuple{4, AbstractArray},  Γ::AbstractMatrix, Γbar::AbstractMatrix, t₀::Integer,  p::AbstractVector)

Contract the perambulators `τ₁_αkβlt` and `τ₂_αkβlt` and the sparse Laplace modes stored in
`sparse_modes_arrays` to get the connected meson correlator where `τ₁_αkβlt` is used to
propagate in forward and `τ₂_αkβlt` in backward direction. The matrices `Γ` and `Γbar` are
the matrices in the interpolating operators. This gives a vacuum expectation value of the
form \
`<(ψ₂Γψ₁)(x) (ψ₁Γbarψ₂)(0)>` \
(in position space) where the indices of the `ψ`'s match the indiced of the perambulators.

The result is store it in `Cₜ`. The source time `t₀` is used to circularly shift `Cₜ` such
that the source time is at the origin. The array `p` is the integer momentum that is used
for the momentum projection of the correlator.
"""
function meson_connected_sparse_contraction!(
    Cₜ::AbstractVector, τ₁_αkβlt::AbstractArray, τ₂_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray},  Γ::AbstractMatrix, Γbar::AbstractMatrix,
    t₀::Integer, p::AbstractVector
)
    # Multiply γ₅ to Gammas
    γ₅Γ = γ[5]*Γ
    Γbarγ₅ = -Γbar*γ[5]

    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time `t₀`
    i_t₀ = t₀+1
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Source position and Laplace modes at source time `t₀`
    x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

    # Compute exp(±ipx) and reshape it to match shape of Laplace modes
    exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./parms.Nₖ)'*p)
    exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))
    exp_ipx_src_iₓ = exp.(2π*im * (x_src_μiₓ_t₀./parms.Nₖ)'*p)
    exp_ipx_src_iₓ = reshape(exp_ipx_src_iₓ, (1, N_points, 1))
    
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulators and Laplace modes at sink time t (index `iₜ`)
        τ₁_αkβl_t = @view τ₁_αkβlt[:, :, :, :,iₜ]
        τ₂_αkβl_t = @view τ₂_αkβlt[:, :, :, :,iₜ]
        v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = conj(v_sink_ciₓk_t[a, iₓ', k]) * 
                (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ', k'] *
                γ₅Γ[α, α'] * τ₁_αkβl_t[α', k', β, l'] *
                conj(v_src_ciₓk_t₀[b, iₓ, l']) *
                (exp_ipx_src_iₓ .* v_src_ciₓk_t₀)[b, iₓ, l] *
                Γbarγ₅[β, β'] * conj(τ₂_αkβl_t[α, k, β', l])
        end

        # Normalization
        C *= (prod(parms.Nₖ)/N_points)^2

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end
