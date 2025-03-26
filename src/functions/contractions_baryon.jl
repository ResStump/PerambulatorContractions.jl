@doc raw"""
    baryon_contractions(τ₁_αkβl_t::AbstractArray, τ₂_αkβl_t::AbstractArray, τ₃_αkβl_t::AbstractArray, Φ_Kiₚ_t::AbstractArray, Φ_Kiₚ_t₀::AbstractArray, Γ_tuple_arr::AbstractVector{<:AbstractVector{<:AbstractMatrix}}, iₚ::Integer, flavour::String, trev::Bool=false) -> C_tn

Contract the perambulators `τ₁_αkβl_t`, `τ₂_αkβl_t` and `τ₃_αkβl_t`, and the mode triplets
`Φ_Kiₚ_t` and `Φ_Kiₚ_t₀` to get baryon correlators. These arrays are assumed to only contain
data for a single sink time `t` and source time `t₀`. The matrices in
`Γ_tuple_arr = [[Γ_A₁, Γ_B₁], [Γ_A₂, Γ_B₂], ...]` are the matrices in
the interpolating operators `O_n = ϵ_abc Γ_Aₙ ψ₁_a (ψ₂_b^T Γ_Bₙ ψ₃_c)`.
For each index `n` compute the correlator `C_tn = <tr[O_n(t, p) O_n(t₀, p)^†]>`.

The momentum index `iₚ` is used for the momentum projection and `flavour` specifies the
flavour content of the baryon. The supported flavours are `"uds"` and `"uud"`.

If `trev=true` (default) apply Euclidean time reversal.
"""
function baryon_contractions(
    τ₁_αkβl_t::AbstractArray, τ₂_αkβl_t::AbstractArray, τ₃_αkβl_t::AbstractArray,
    Φ_Kiₚ_t::AbstractArray, Φ_Kiₚ_t₀::AbstractArray,
    Γ_tuple_arr::AbstractVector{<:AbstractVector{<:AbstractMatrix}}, iₚ::Integer,
    flavour::String, trev::Bool=false
)
    # Number of gamma matrix tuples
    Nᵧ = length(Γ_tuple_arr)

    # Apply Euclidean time reversal
    if trev
        Γ_tuple_arr = deepcopy(Γ_tuple_arr)
        for (i, Γ_tuple) in enumerate(Γ_tuple_arr)
            Γ_tuple_arr[i][1] = Γ_tuple[1] * γ[4]*γ[5]
            Γ_tuple_arr[i][2] = transpose(γ[4]*γ[5]) * Γ_tuple[2] * γ[4]*γ[5]
        end
    end

    # Gamma matrices in adjoint operator
    Γbar_tuple_arr = deepcopy(Γ_tuple_arr)
    for (i,  Γ_tuple) in enumerate(Γ_tuple_arr)
        Γbar_tuple_arr[i][1] = γ[4] * Γ_tuple[1]'
        Γbar_tuple_arr[i][2] = γ[4] * Γ_tuple[2]' * γ[4]
    end

    # Sign flip in adjoint operator for Euclidean time reversal
    if trev
        for i in eachindex(Γbar_tuple_arr)
            Γbar_tuple_arr[i][1] *= -1
        end
    end

    # Check flavour
    if flavour ∉ ["uds", "uud"]
        throw(ArgumentError("unsupported flavour."))
    end

    # Initialize ITensors
    #####################

    # Indices
    N_modes = size(τ₁_αkβl_t, 2)
    α = IT.Index(4, "α")
    β = IT.Index(4, "β")
    γ_ = IT.Index(4, "γ")
    δ = IT.Index(4, "δ")
    k = IT.Index(N_modes, "k")
    l = IT.Index(N_modes, "l")
    h = IT.Index(N_modes, "h")

    # Mode triplets at sink (as dense tensors)
    Φ_klh_tiₚ = antisym_to_dense(@view(Φ_Kiₚ_t[:, iₚ]))
    Φ_t = IT.itensor(Φ_klh_tiₚ, k', l', h')

    # Mode triplets at source  (as dense tensors)
    Φ_klh_t₀iₚ = antisym_to_dense(@view(Φ_Kiₚ_t₀[:, iₚ]))
    Φ_t₀ = IT.itensor(Φ_klh_t₀iₚ, k, l, h)

    # Perambulators
    τ₁_t = IT.itensor(τ₁_αkβl_t, α', k', α, k)
    τ₂_t = IT.itensor(τ₂_αkβl_t, β', l', β, l)
    τ₃_t = IT.itensor(τ₃_αkβl_t, γ_', h', γ_, h)

    # Tensor contractions
    #####################

    # Precontraction
    T = (Φ_t * τ₁_t) * (conj(Φ_t₀) * τ₂_t) * τ₃_t

    # Initialize array for diagonal correlator matrix elements
    C_n = zeros(ComplexF64, Nᵧ)

    # Loop over all gamma matrices
    for n in 1:Nᵧ
        Γ_A = IT.itensor(Γ_tuple_arr[n][1], δ, α')
        Γ_B = IT.itensor(Γ_tuple_arr[n][2], β', γ_')
        Γbar_A = IT.itensor(Γbar_tuple_arr[n][1], α, δ)
        Γbar_B = IT.itensor(Γbar_tuple_arr[n][2], γ_, β)

        if flavour in ["uds", "uud"]
            C_n[n] += IT.scalar(T * (Γ_B * Γbar_B) * (Γ_A * Γbar_A))
        end
        if flavour in ["uud"]
            T_ = IT.swapind(T, α, β)
            C_n[n] += IT.scalar(T_ * (Γ_B * Γbar_B) * (Γ_A * Γbar_A))
        end
    end

    return C_n
end
