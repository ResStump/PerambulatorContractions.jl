module Startup

include("../../src/IO.jl")
include("../../src/contractions.jl")
import PrecompileTools as PT

PT.@setup_workload begin
    # Set parameters
    parms_toml_string = ""
    perambulator_dir, mode_doublets_dir, sparse_modes_dir, result_dir = "", "", "", "" 
    cnfg_indices, tsrc_arr = zeros(Int, 1), zeros(Int, 1, 1)
    Nₜ, Nₖ, N_modes = 16, [8, 8, 8], 10
    N_cnfg, N_src = 1, 1
    p = [1, 0, 0]

    parms = Parms(parms_toml_string, perambulator_dir, mode_doublets_dir,
                  sparse_modes_dir, result_dir, cnfg_indices, tsrc_arr, Nₜ, Nₖ,
                  N_modes, N_cnfg, N_src, p)
    
    # Pseudoscalar contraction
    N_points = 100

    # Chose some time and momentum index
    t₀, iₚ = 1, 2
    
    # Mode doublets
    Φ_kltiₚ = rand(ComplexF64, N_modes, N_modes, Nₜ, 10)

    # Perambulator
    τ_αkβlt = rand(ComplexF64, 4, N_modes, 4, N_modes, Nₜ)

    # Sparse modes
    x_sink_μiₓ = rand(ComplexF64, 3, N_points)
    x_src_μiₓt = rand(ComplexF64, 3, N_points, Nₜ)
    v_sink_ciₓkt = rand(ComplexF64, 3, N_points, N_modes, Nₜ)
    v_src_ciₓkt = rand(ComplexF64, 3, N_points, N_modes, Nₜ)
    sparse_modes_arrays = x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

    #= # exp(±ipx)
    exp_mipx_sink_iₓ = rand(ComplexF64, 1, N_points, 1)
    exp_ipx_src_iₓ = rand(ComplexF64, 1, N_points, 1) =#

    # Correlator
    Cₜ = Vector{ComplexF64}(undef, parms.Nₜ)


    # Laplace modes at source time 't₀'
    #= v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀] =#

    #= τ₁ = rand(ComplexF64, 4, N_modes, 4, N_modes)
    τ₂ = rand(ComplexF64, 4, N_modes, 4, N_modes)
    τ₁_view = @view τ₁[:, :, :, :]
    τ₂_view = @view τ₂[:, :, :, :]

    Φ₁ = rand(ComplexF64, N_modes, N_modes)
    Φ₂ = rand(ComplexF64, N_modes, N_modes)
    Φ₁_view = @view Φ₁[:, :]
    Φ₂_view = @view Φ₂[:, :] =#

    PT.@compile_workload begin
        pseudoscalar_contraction_p0!(Cₜ, τ_αkβlt, t₀)
        pseudoscalar_contraction!(Cₜ, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
        pseudoscalar_sparse_contraction!(Cₜ, τ_αkβlt, sparse_modes_arrays, t₀, p)

        # Tensor contractions
        #= TO.@tensoropt begin
            C = τ₁[α, k, β, l] * conj(τ₂[α, k, β, l])
        end
        TO.@tensoropt begin
            C = Φ₁[k, k'] * τ₁[α, k', β, l'] *
                conj(Φ₂[l, l']) * conj(τ₂[α, k, β, l])
        end =#
        #= TO.@tensoropt begin
            C = τ₁_view[α, k, β, l] * conj(τ₂_view[α, k, β, l])
        end
        TO.@tensoropt begin
            C = Φ₁_view[k, k'] * τ₁_view[α, k', β, l'] *
                conj(Φ₂_view[l, l']) * conj(τ₂_view[α, k, β, l])
        end =#

        #= # Mode doublet and Laplace modes at source time 't₀'
        Φ_kl_t₀iₚ = @view Φ_kltiₚ[:, :, i_t₀, iₚ]
        v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

        # Loop over all sink time indice
        for iₜ in 1:Nₜ
            # Mode doublet, perambulator and Laplace modes at sink time t (index 'iₜ')
            Φ_kl_tiₚ = @view Φ_kltiₚ[:, :, iₜ, iₚ]
            τ_αkβl_t = @view τ_αkβlt[:, :, :, :,iₜ]
            v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

            # Tensor contractions
            TO.@tensoropt begin
                C = conj(v_sink_ciₓk_t[a, iₓ', k]) * 
                    (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ', k'] *
                    τ_αkβl_t[α, k', β, l'] *
                    conj(v_src_ciₓk_t₀[b, iₓ, l']) *
                    (exp_ipx_src_iₓ .* v_src_ciₓk_t₀)[b, iₓ, l] *
                    conj(τ_αkβl_t[α, k, β, l])
            end
            TO.@tensoropt begin
                C = Φ_kl_tiₚ[k, k'] * τ_αkβl_t[α, k', β, l'] *
                    conj(Φ_kl_t₀iₚ[l, l']) * conj(τ_αkβl_t[α, k, β, l])
            end
            TO.@tensoropt begin
                C = τ_αkβl_t[α, k, β, l] * conj(τ_αkβl_t[α, k, β, l])
            end
        end =#
    end
end

end # module Startup
