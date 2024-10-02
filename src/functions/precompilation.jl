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
                  Nₜ, Nₖ, N_modes, N_cnfg, N_src, [1], p_arr)
    
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

    # Meson correlator
    Cₜ = Vector{ComplexF64}(undef, parms.Nₜ)

    # Meson-meson correlator
    correlator_size = (parms.Nₜ, 1, 1, 1, 1, length(parms.p_arr))
    C_tnmn̄m̄iₚ = Array{ComplexF64}(undef, correlator_size)
    C_tnmn̄m̄ = @view C_tnmn̄m̄iₚ[:, :, :, :, 1]

    # Correlator with only two gamma matrix indices
    C_tnmiₚ = Array{ComplexF64}(undef, parms.Nₜ, 1, 1, length(parms.p_arr))

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
        DD_local_contractons!(C_tnmn̄m̄iₚ, τ_αkβlt, τ_αkβlt, sparse_modes_arrays,
                              [Γ], t₀, p_arr)
        DD_nonlocal_contractons!(C_tnmn̄m̄, τ_αkβlt, τ_αkβlt, Φ_kltiₚ, [Γ], t₀,
                                 [iₚ, iₚ, iₚ, iₚ])
        DD_mixed_contractons!(C_tnmn̄m̄iₚ, C_tnmn̄m̄iₚ, τ_αkβlt, τ_αkβlt, Φ_kltiₚ,
                              sparse_modes_arrays, [Γ], t₀, [iₚ, iₚ], p_arr)
        dad_local_contractons!(C_tnmiₚ, τ_αkβlt, τ_αkβlt, sparse_modes_arrays,
                              [Γ], [Γ], t₀, p_arr)
        DD_dad_nonlocal_local_mixed_contractons!(C_tnmn̄m̄iₚ, C_tnmn̄m̄iₚ, τ_αkβlt, τ_αkβlt,
                                                 Φ_kltiₚ, sparse_modes_arrays,
                                                 [Γ], [Γ], [Γ], t₀, [iₚ, iₚ], p_arr)
        DD_dad_local_mixed_contractons!(C_tnmn̄m̄iₚ, C_tnmn̄m̄iₚ, τ_αkβlt, τ_αkβlt,
                                        sparse_modes_arrays, [Γ], [Γ], [Γ], t₀, p_arr)
    end
end