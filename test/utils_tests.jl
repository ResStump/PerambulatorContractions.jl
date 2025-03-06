# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

@testset "Test parse_gamma_string" begin
    Gamma = ["I", "1", "G1", "G2", "G3", "G4", "G5",
             "G1G5", "G2G5", "G3G5", "G4G5",
             "sigma12" , "sigma13", "sigma14", "sigma23", "sigma24", "sigma34"]
    Γ = [PC.I, PC.I, PC.γ[1], PC.γ[2], PC.γ[3], PC.γ[4], PC.γ[5],
        PC.γ[1]*PC.γ[5], PC.γ[2]*PC.γ[5], PC.γ[3]*PC.γ[5], PC.γ[4]*PC.γ[5],
        PC.σ_μν(1, 2), PC.σ_μν(1, 3), PC.σ_μν(1, 4), PC.σ_μν(2, 3), PC.σ_μν(2, 4),
        PC.σ_μν(3, 4)]

    @test PC.parse_gamma_string.(Gamma) == Γ
    @test PC.parse_gamma_string.("+" .* Gamma) == Γ
    @test PC.parse_gamma_string.("-" .* Gamma) == -Γ
    @test PC.parse_gamma_string.("i" .* Gamma) == im*Γ
    @test PC.parse_gamma_string.("C" .* Gamma) == [PC.C].*Γ
    @test PC.parse_gamma_string.("iC" .* Gamma) == im*[PC.C].*Γ
    @test PC.parse_gamma_string.("-iC" .* Gamma) == -im*[PC.C].*Γ
end

# Paths to files
sparse_modes_file_Nsep1 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"
sparse_modes_file_Nsep2 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"

# Read files
p_arr = PC.read_mode_doublet_momenta(mode_doublets_file)
sparse_modes_arrays_Nsep1 = PC.read_sparse_modes(sparse_modes_file_Nsep1)
sparse_modes_arrays_Nsep2 = PC.read_sparse_modes(sparse_modes_file_Nsep2)
Φ_kltiₚ = PC.read_mode_doublets(mode_doublets_file)


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

@testset "Test sparse_mode_doublets" begin
    iₚ_arr = collect(eachindex(p_arr))
    Φ_sink_kltiₚ = similar(Φ_kltiₚ)
    Φ_src_kltiₚ = similar(Φ_kltiₚ)

    PC.sparse_mode_doublets!(Φ_sink_kltiₚ, Φ_src_kltiₚ, sparse_modes_arrays_Nsep1,
                             iₚ_arr, p_arr)

    @test Φ_kltiₚ ≈ Φ_sink_kltiₚ
    @test Φ_kltiₚ ≈ Φ_src_kltiₚ
end
