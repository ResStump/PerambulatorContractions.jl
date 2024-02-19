import MKL
import LinearAlgebra as LA
import TensorOperations as TO

@doc raw"""
    pseudoscalar_contraction_p0!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray, t₀::Integer)

Contract the perambulator 'τ\_αkβlt' to get the pseudoscalar correlator and 
store it in 'Cₜ'. The source time 't₀' is used to circularly shift 'Cₜ' such that the
source time is at the origin.
"""
function pseudoscalar_contraction_p0!(
    Cₜ::AbstractVector, τ_αkβlt::AbstractArray, t₀::Integer
)
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulator at sink (index 'iₜ')
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]

        # Tensor contraction
        TO.@tensoropt begin
            C = τ_αkβl_t[α, k, β, l] * conj(τ_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    pseudoscalar_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                              Φ_kltiₚ::AbstractArray,  t₀::Integer, iₚ::Integer)

Contract the perambulator 'τ\_αkβlt' and the mode doublet 'Φ\_kltiₚ' to get the
pseudoscalar correlator and store it in 'Cₜ'. The source time 't₀' is used to circularly
shift 'Cₜ' such that the source time is at the origin. The index 'iₚ' sets the momentum
that is used from the mode doublet.
"""
function pseudoscalar_contraction!(
    Cₜ::AbstractVector, τ_αkβlt::AbstractArray, Φ_kltiₚ::AbstractArray,
    t₀::Integer, iₚ::Integer
)
    # Index for source time 't₀'
    i_t₀ = mod1(t₀+1, parms.Nₜ)

    # Mode doublet at source time 't₀'
    Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]

    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Mode doublet and perambulator at sink time t (index 'iₜ')
        Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :,iₜ]
        
        # Tensor contraction
        TO.@tensoropt begin
            C = Φ_kl_tiₚ[k, k'] * τ_αkβl_t[α, k', β, l'] *
                conj(Φ_kl_t₀iₚ[l, l']) * conj(τ_αkβl_t[α, k, β, l])
        end

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end

@doc raw"""
    pseudoscalar_sparse_contraction!(Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
                                     sparse_modes_arrays::NTuple{4, AbstractArray},
                                     t₀::Integer, p::Vector)

Contract the perambulator 'τ\_αkβlt' and the sparse Laplace modes stored in
'sparse\_modes\_arrays' to get the pseudoscalar correlator and store it in 'Cₜ'. The source
time 't₀' is used to circularly shift 'Cₜ' such that the source time is at the origin.
The array 'p' is the integer momentum that is used for the momentum projection of the
correlator.
"""
function pseudoscalar_sparse_contraction!(
    Cₜ::AbstractVector, τ_αkβlt::AbstractArray,
    sparse_modes_arrays::NTuple{4, AbstractArray}, t₀::Integer, p::Vector
)
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Index for source time 't₀'
    i_t₀ = mod1(t₀+1, parms.Nₜ)
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Laplace modes at source time 't₀'
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]
    
    # Loop over all sink time indice
    for iₜ in 1:parms.Nₜ
        # Perambulator and Laplace modes at sink time t (index 'iₜ')
        τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]
        v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

        # Source position for sink time t
        x_src_μiₓ_t = @view x_src_μiₓt[:, :, iₜ]

        # Compute exp(±ipx) and reshape it to match shape of Laplace modes
        exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./parms.Nₖ)'*p)
        exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))
        exp_ipx_src_iₓ = exp.(2π*im * (x_src_μiₓ_t./parms.Nₖ)'*p)
        exp_ipx_src_iₓ = reshape(exp_ipx_src_iₓ, (1, N_points, 1))

        # Tensor contraction
        TO.@tensoropt begin
            C = conj(v_sink_ciₓk_t[a, iₓ', k]) * 
                (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ', k'] *
                τ_αkβl_t[α, k', β, l'] *
                conj(v_src_ciₓk_t₀[b, iₓ, l']) *
                (exp_ipx_src_iₓ .* v_src_ciₓk_t₀)[b, iₓ, l] *
                conj(τ_αkβl_t[α, k, β, l])
        end

        # Normalization
        C *= (prod(parms.Nₖ)/N_points)^2

        # Circularly shift time such that t₀=0
        Cₜ[mod1(iₜ-t₀, parms.Nₜ)] = C
    end

    return
end