PrecompileTools.@setup_workload begin
    # Set parameters
    parms_toml_string = ""
    perambulator_dir, perambulator_charm_dir, mode_doublets_dir = "", "", ""
    sparse_modes_dir, result_dir = "", ""
    cnfg_indices, tsrc_arr = zeros(Int, 1), zeros(Int, 1, 1)
    Nₜ, Nₖ, N_modes = 16, [8, 8, 8], 10
    N_cnfg, N_src = 1, 1
    p_arr = [[1, 0, 0]]

    parms = Parms(parms_toml_string, perambulator_dir, perambulator_charm_dir,
                  mode_doublets_dir, sparse_modes_dir, result_dir, cnfg_indices, tsrc_arr,
                  Nₜ, Nₖ, N_modes, N_cnfg, N_src, p_arr)
    
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

    # Correlator
    Cₜ = Vector{ComplexF64}(undef, parms.Nₜ)

    # Matrices in interpolstors
    Γ, Γbar = γ[5], -γ[5]

    PrecompileTools.@compile_workload begin
        meson_connected_contraction_p0!(Cₜ, τ_αkβlt, τ_αkβlt, Γ, Γbar, t₀)
        meson_connected_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ, Γbar, t₀, iₚ)
        meson_connected_sparse_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, sparse_modes_arrays,
                                            Γ, Γbar, t₀, p_arr[1])
    end
end