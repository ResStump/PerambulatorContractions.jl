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
    
    # Pseudoscalar contraction
    N_points = 100

    # Chose some time and momentum index
    t₀, iₚ = 1, 2
    
    # Mode doublets
    Φ_kltiₚ = rand(ComplexF64, N_modes, N_modes, Nₜ, 10)

    # Perambulator
    τ_αkβlt = rand(ComplexF64, 4, N_modes, 4, N_modes, Nₜ)

    # Sparse modes
    x_sink_μiₓt = rand(Int32, 3, N_points, Nₜ)
    x_src_μiₓt = rand(Int32, 3, N_points, Nₜ)
    v_sink_ciₓkt = rand(ComplexF64, 3, N_points, N_modes, Nₜ)
    v_src_ciₓkt = rand(ComplexF64, 3, N_points, N_modes, Nₜ)
    sparse_modes_arrays = x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt

    # Select sink time `t=11`
    τ_αkβl_t = @view τ_αkβlt[:, :, :, :, 11]
    Φ_kliₚ_t = @view Φ_kltiₚ[:, :, 11, :]
    x_sink_μiₓ_t = @view x_sink_μiₓt[:, :, 11]
    v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, 11]

    # Select source time `t₀=0`
    Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, 1, :]
    x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, 1]
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, 1]

    sparse_modes_arrays_tt₀ = x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀

    # Meson correlator
    Cₜ = Vector{ComplexF64}(undef, parms.Nₜ)

    # Matrices in interpolstors
    Γ, Γbar = γ[5], -γ[5]

    PrecompileTools.@compile_workload begin
        pseudoscalar_contraction_p0!(Cₜ, τ_αkβlt, t₀)
        pseudoscalar_contraction!(Cₜ, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
        pseudoscalar_sparse_contraction!(Cₜ, τ_αkβlt, sparse_modes_arrays, t₀, p_arr[1])
        meson_connected_contraction_p0!(Cₜ, τ_αkβlt, τ_αkβlt, Γ, Γbar, t₀)
        meson_connected_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, Γ, Γbar, t₀, iₚ)
        meson_connected_sparse_contraction!(Cₜ, τ_αkβlt, τ_αkβlt, sparse_modes_arrays,
                                            Γ, Γbar, t₀, p_arr[1])
        DD_local_contractons(τ_αkβl_t, τ_αkβl_t, sparse_modes_arrays_tt₀, [Γ], p_arr)
        DD_nonlocal_contractons(τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀, [Γ],
                                [iₚ, iₚ, iₚ, iₚ])
        DD_mixed_contractons(τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀,
                             sparse_modes_arrays_tt₀, [Γ], [iₚ, iₚ], p_arr)
        dad_local_contractons(τ_αkβl_t, τ_αkβl_t, sparse_modes_arrays_tt₀, [Γ], [Γ], p_arr)
        DD_dad_nonlocal_local_mixed_contractons(τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀,
                                                sparse_modes_arrays_tt₀,
                                                [Γ], [Γ], [Γ], [iₚ, iₚ], p_arr)
        DD_dad_local_mixed_contractons(τ_αkβl_t, τ_αkβl_t, sparse_modes_arrays_tt₀,
                                       [Γ], [Γ], [Γ], p_arr)
    end
end