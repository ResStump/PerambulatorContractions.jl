# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/16x8v1.toml")

# Read parameters from infile
PC.read_parameters(:meson)


# Paths to files
perambulator_file = PC.parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
sparse_modes_file_Nsep1 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"
sparse_modes_file_Nsep2 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"

# Set momenta and their indices
p_arr = [[0, 0, 0], [1, 0, 0]]
iₚ_arr = [findfirst(p_ -> p_ == p, PC.parms.p_arr) for p in p_arr]

# Read files
τ_αkβlt = PC.read_perambulator(perambulator_file)
Φ_kltiₚ = PC.read_mode_doublets(mode_doublets_file)
sparse_modes_arrays_Nsep1 = PC.read_sparse_modes(sparse_modes_file_Nsep1)
sparse_modes_arrays_Nsep2 = PC.read_sparse_modes(sparse_modes_file_Nsep2)

@testset "Connected meson contraction tests" begin
    # Gamma matrices
    Γ_arr = [PC.γ[5], PC.γ[1]]
    Γbar_arr = [PC.γ[4]] .* adjoint.(Γ_arr) .* [PC.γ[4]]
    Nᵧ = length(Γ_arr)

    # Generate contraction functions for sparse contractions
    contract_arr = Matrix{Function}(undef, Nᵧ, Nᵧ)
    for n in 1:Nᵧ, n̄ in 1:Nᵧ
        contract_arr[n, n̄] = PC.generate_meson_connected_contract_func(
            Γ_arr[n], Γbar_arr[n̄]
        )
    end

    # Choose one sink time `t`
    t = 10
    iₜ = t + 1

    # Source time t₀ and its index
    t₀ = PC.parms.tsrc_arr[1, 1]
    i_t₀ = t₀+1

    # Compare correlator from mode doublets and from full Laplace modes to it self and to
    # saved correlators

    # Unpack sparse modes arrays
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays_Nsep1

    # Select source time `t₀`
    Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]
    x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
    v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

    # Views of tensors
    τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]
    Φ_kliₚ_t = @view Φ_kltiₚ[:, :, iₜ, :]
    x_sink_μiₓ_t = @view x_sink_μiₓt[:, :, iₜ]
    v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

    # Compute correlator from mode doublets
    C_nn̄iₚ_t_full = Array{ComplexF64}(undef, 2, 2, length(p_arr))
    for (i_p, iₚ) in enumerate(iₚ_arr)
        C_nn̄iₚ_t_full[:, :, i_p] = PC.meson_connected_contractions(
            τ_αkβl_t, τ_αkβl_t, Φ_kliₚ_t, Φ_kliₚ_t₀, Γ_arr, iₚ, true
        )
    end

    # Compute correlator from full modes
    C_nn̄iₚ_t_modes = PC.meson_connected_sparse_contractions(
        τ_αkβl_t, τ_αkβl_t, (x_sink_μiₓ_t, x_src_μiₓ_t₀, v_sink_ciₓk_t, v_src_ciₓk_t₀),
        p_arr, contract_arr, true
    )

    @test C_nn̄iₚ_t_full ≈ C_nn̄iₚ_t_modes

    @test C_nn̄iₚ_t_full[:, :, 1] ≈
        PC.HDF5.h5read("16x8v1_data/16x8v1_6modes_meson_connected_llbar_full.hdf5",
                       "Correlators/p0,0,0")[iₜ-t₀, :, :, 1, 1]
    @test C_nn̄iₚ_t_full[:, :, 2] ≈
        PC.HDF5.h5read("16x8v1_data/16x8v1_6modes_meson_connected_llbar_full.hdf5",
                       "Correlators/p1,0,0")[iₜ-t₀, :, :, 1, 1]

    @test C_nn̄iₚ_t_modes[:, :, 1] ≈
        PC.HDF5.h5read("16x8v1_data/16x8v1_6modes_meson_connected_llbar_sparse_Nsep1.hdf5",
                       "Correlators/p0,0,0")[iₜ-t₀, :, :, 1, 1]
    @test C_nn̄iₚ_t_modes[:, :, 2] ≈
        PC.HDF5.h5read("16x8v1_data/16x8v1_6modes_meson_connected_llbar_sparse_Nsep1.hdf5",
                       "Correlators/p1,0,0")[iₜ-t₀, :, :, 1, 1]
end
