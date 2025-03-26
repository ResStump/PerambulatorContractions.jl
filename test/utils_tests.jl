# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/16x8v1.toml")

# Read parameters from infile
PC.read_parameters(:baryon)

@testset "Test parse_gamma_string" begin
    Gamma = ["I", "G1", "G2", "G3", "G4", "G5",
             "sigma12" , "sigma13", "sigma14", "sigma23", "sigma24", "sigma34",
             "C", "Pp", "Pm"]
    Γ = [PC.I, PC.γ[1], PC.γ[2], PC.γ[3], PC.γ[4], PC.γ[5],
         PC.σ_μν(1, 2), PC.σ_μν(1, 3), PC.σ_μν(1, 4), PC.σ_μν(2, 3), PC.σ_μν(2, 4),
         PC.σ_μν(3, 4),
         PC.C, PC.Pp, PC.Pm]

    # Single matrix with prefactor
    for (prefactor, prefactor_str) in zip([1, 1, -1, im, -im], ["", "+", "-", "i", "-i"])
        @test PC.parse_gamma_string.(prefactor_str .* Gamma) == prefactor .* Γ
    end

    # Products of two matrices
    for ((Γ₁, Gamma1), (Γ₂, Gamma2)) in Iterators.product(zip(Γ, Gamma), zip(Γ, Gamma))
        @test PC.parse_gamma_string("$(Gamma1)*$(Gamma2)") == Γ₁ * Γ₂
        @test PC.parse_gamma_string("$(Gamma1) * $(Gamma2)") == Γ₁ * Γ₂
    end

    # Products of three matrices
    for ((Γ₁, Gamma1), (Γ₂, Gamma2), (Γ₃, Gamma3)) in Iterators.product(zip(Γ, Gamma),
                                                                        zip(Γ, Gamma),
                                                                        zip(Γ, Gamma))
        @test PC.parse_gamma_string("$(Gamma1)*$(Gamma2)*$(Gamma3)") == Γ₁ * Γ₂ * Γ₃
    end
end

# Paths to files
perambulator_file = PC.parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
sparse_modes_file_Nsep1 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"
sparse_modes_file_Nsep2 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
mode_triplets_file = PC.parms.mode_triplets_dir/"mode_triplets_16x8v1n1"

# Read files
τ_αkβlt = PC.read_perambulator(perambulator_file)
p_arr = PC.read_mode_doublets_momenta(mode_doublets_file)
@assert p_arr == PC.read_mode_triplets_momenta(mode_doublets_file)
sparse_modes_arrays_Nsep1 = PC.read_sparse_modes(sparse_modes_file_Nsep1)
sparse_modes_arrays_Nsep2 = PC.read_sparse_modes(sparse_modes_file_Nsep2)
Φ_kltiₚ = PC.read_mode_doublets(mode_doublets_file)
Φ_Ktiₚ = PC.read_mode_triplets(mode_triplets_file)

@testset "Test increase_separation" begin
    sparse_modes_arrays_new_Nsep2 = PC.increase_separation(sparse_modes_arrays_Nsep1, 2)

    @test sparse_modes_arrays_new_Nsep2 == sparse_modes_arrays_Nsep2
end

@testset "Compare full modes at sink and src" begin
    x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays_Nsep1

    # Loop over all times
    for iₜ in 1:PC.parms.Nₜ
        # Loop over all positions at sink
        for (iₓ_sink, x_sink) in enumerate(eachcol(x_sink_μiₓt[:, :, iₜ]))
            # Find position index at source that corresponds to that at sink
            iₓ_src = findfirst(x -> x == x_sink, eachcol(x_src_μiₓt[:, :, iₜ]))

            @test v_sink_ciₓkt[:, iₓ_sink, :, iₜ] == v_src_ciₓkt[:, iₓ_src, :, iₜ]
        end
    end
end

@testset "Test sparse_mode_doublets!" begin
    iₚ_arr = collect(eachindex(p_arr))
    Φ_sink_kltiₚ = similar(Φ_kltiₚ)
    Φ_src_kltiₚ = similar(Φ_kltiₚ)

    PC.sparse_mode_doublets!(Φ_sink_kltiₚ, Φ_src_kltiₚ, sparse_modes_arrays_Nsep1,
                             iₚ_arr, p_arr)

    @test Φ_kltiₚ ≈ Φ_sink_kltiₚ
    @test Φ_kltiₚ ≈ Φ_src_kltiₚ
end

@testset "Test sparse_mode_triplets!" begin
    a, b, c = rand(3), rand(3), rand(3)
    @test PC.scalar_triple_product(a, b, c) ≈ LA.dot(a, LA.cross(b, c))

    iₚ_arr = collect(eachindex(p_arr))
    Φ_sink_Ktiₚ = similar(Φ_Ktiₚ)
    Φ_src_Ktiₚ = similar(Φ_Ktiₚ)

    PC.sparse_mode_triplets!(Φ_sink_Ktiₚ, Φ_src_Ktiₚ, sparse_modes_arrays_Nsep1,
                             iₚ_arr, p_arr)

    @test Φ_Ktiₚ ≈ Φ_sink_Ktiₚ
    @test Φ_Ktiₚ ≈ Φ_src_Ktiₚ
end

@testset "Test antisym functions" begin
    iₜ, iₚ = 1, 1
    Φ_K_tiₚ = @view Φ_Ktiₚ[:, iₜ, iₚ]
    τ_αkβl_t = @view τ_αkβlt[:, :, :, :, iₜ]

    # Sparse calculation
    Φτ_αβklh = PC.antisym_contraction(Φ_K_tiₚ, τ_αkβl_t, 2)

    α = IT.Index(4, "α")
    β = IT.Index(4, "β")
    k = IT.Index(PC.parms.N_modes, "k")
    l = IT.Index(PC.parms.N_modes, "l")
    h = IT.Index(PC.parms.N_modes, "h")

    Φ_klh = PC.antisym_to_dense(Φ_K_tiₚ)

    Φ = IT.itensor(Φ_klh, k', l, h)
    τ = IT.itensor(τ_αkβl_t, α, k', β, k)

    # Dense calculation
    Φτ = Φ * τ

    @test Φτ_αβklh ≈ IT.array(Φτ, α, β, k, l, h)
end
