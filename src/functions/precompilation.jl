PrecompileTools.@setup_workload begin
    # Set parameters
    parms_toml_string = ""
    perambulator_dir, perambulator_charm_dir, mode_doublets_dir = "", "", ""
    sparse_modes_dir, result_dir = "", ""
    cnfg_numbers, tsrc_arr = zeros(Int, 1), zeros(Int, 1, 1)
    Nₜ, Nₖ, N_modes = 16, [8, 8, 8], 10
    N_cnfg, N_src = 1, 1
    p_arr = [[1, 0, 0]]
    N_ranks_per_cnfg = 1

    parms = Parms(parms_toml_string, perambulator_dir, perambulator_charm_dir,
                  mode_doublets_dir, sparse_modes_dir, result_dir, cnfg_numbers, tsrc_arr,
                  Nₜ, Nₖ, N_modes, N_cnfg, N_src, [1], p_arr, N_ranks_per_cnfg)
    
    # Number of points on spares lattice
    N_points = 8

    # Choose some time and momentum index
    t₀, iₚ = 1, 2
    
    # Mode doublets
    Φ_kliₚ_t = rand(ComplexF64, N_modes, N_modes, 2)
    Φ_kliₚ_t₀ = rand(ComplexF64, N_modes, N_modes, 2)

    # Perambulator
    τ_αkβl_t = rand(ComplexF64, 4, N_modes, 4, N_modes)

    # Sparse modes
    x_sink_μiₓ_t = rand(Int32, 3, N_points)
    x_src_μiₓ_t₀ = rand(Int32, 3, N_points)
    v_sink_ciₓk_t = rand(ComplexF64, 3, N_points, N_modes)
    v_src_ciₓk_t₀ = rand(ComplexF64, 3, N_points, N_modes)
    sparse_modes_arrays_tt₀ = x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀

    # Matrices in interpolstors
    Γ, Γbar = γ[5], -γ[5]

    PrecompileTools.@compile_workload begin
        # Meson contractions
        meson_connected_contractions(τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀, [Γ], iₚ,
                                     false)
        meson_connected_contractions(τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀, [Γ], iₚ, true)
        #= contract = generate_meson_connected_contract_func(Γ, Γbar)
        meson_connected_sparse_contractions(τ_αkβl_t, τ_αkβl_t, sparse_modes_arrays_tt₀,
                                            p_arr, [contract], false) =#
    end
end