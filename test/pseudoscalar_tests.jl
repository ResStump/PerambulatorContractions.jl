# Add infile manually to arguments
pushfirst!(ARGS, "-i", "test/16x8v1_data/pseudoscalar_16x8v1.toml")

# Read parameters from infile
parms, parms_toml = read_parameters()


# Paths to files
perambulator_file = parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
sparse_modes_file = parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"

# Read files
τ_αkβlt = read_perambulator(perambulator_file)
p_arr = read_mode_doublet_momenta(mode_doublets_file)
Φ_kltiₚ = read_mode_doublets(mode_doublets_file)
sparse_modes_arrays = read_sparse_modes(sparse_modes_file)


@testset "Compare full modes at sink and src" begin
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

    # Loop over all positions at sink
    for (iₓ_sink, x_sink) in enumerate(eachcol(x_sink_μiₓ))
        for iₜ in 1:parms.Nₜ
            iₓ_src_t = findfirst(x -> x == x_sink, eachcol(x_src_μiₓt[:, :, iₜ]))

            @test v_sink_ciₓkt[:, iₓ_sink, :, iₜ] == v_src_ciₓkt[:, iₓ_src_t, :, iₜ]
        end
    end

end

@testset "Compare mode doublets from full modes to those from file" begin
    Φ2_kltiₚ = similar(Φ_kltiₚ)
    x_sink_μiₓ, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays
    
    # Number of points on spares lattice
    _, N_points = size(x_sink_μiₓ)

    # Loop over all momenta
    for (iₚ, p) in enumerate(eachrow(p_arr))
        # Loop over all sink time indice
        for iₜ in 1:parms.Nₜ
            # Laplace modes at sink time t (index 'iₜ')
            v_sink_ciₓk_t = @view v_sink_ciₓkt[:, :, :, iₜ]

            # Compute exp(+ipx) and reshape it to match shape of Laplace modes
            exp_mipx_sink_iₓ = exp.(-2π*im * (x_sink_μiₓ./parms.Nₖ)'*p)
            exp_mipx_sink_iₓ = reshape(exp_mipx_sink_iₓ, (1, N_points, 1))

            Φ2_kl_tiₚ = @view Φ2_kltiₚ[:, :, iₜ, iₚ]
            TO.@tensoropt begin
                Φ2_kl_tiₚ[k, l] = conj(v_sink_ciₓk_t[a, iₓ, k]) * 
                    (exp_mipx_sink_iₓ .* v_sink_ciₓk_t)[a, iₓ, l]
            end
        end
    end

    # Check if both mode doublets are the same
    @test Φ_kltiₚ ≈ Φ2_kltiₚ
end

@testset "Compare pseudoscalar correlator from mode doublets and from full modes" begin
    Cₜ_mode_doublets = Array{ComplexF64}(undef, parms.Nₜ)
    Cₜ_mode_doublets_p0 = similar(Cₜ_mode_doublets)
    Cₜ_full_modes = similar(Cₜ_mode_doublets)

    # Source time t₀
    t₀ = parms.tsrc_arr[1, 1]

    # For zero mometum
    pseudoscalar_contraction_p0!(Cₜ_mode_doublets_p0, τ_αkβlt, t₀)

    # Loop over all momenta
    for (iₚ, p) in enumerate(eachrow(p_arr))
        pseudoscalar_contraction!(Cₜ_mode_doublets, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
        pseudoscalar_sparse_contraction!(Cₜ_full_modes, τ_αkβlt, sparse_modes_arrays, t₀, p)

        @test Cₜ_mode_doublets ≈ Cₜ_full_modes

        if p == [0, 0, 0]
            @test Cₜ_mode_doublets ≈ Cₜ_mode_doublets_p0
        end
    end
end
