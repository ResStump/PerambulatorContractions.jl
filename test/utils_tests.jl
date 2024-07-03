# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

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

@testset "Test sparse_mode_doublets" begin
    iₚ_arr = collect(eachindex(p_arr))
    Φ_sink_kltiₚ = similar(Φ_kltiₚ)
    Φ_src_kltiₚ = similar(Φ_kltiₚ)

    PC.sparse_mode_doublets!(Φ_sink_kltiₚ, Φ_src_kltiₚ, sparse_modes_arrays_Nsep1,
                             iₚ_arr, p_arr)

    @test Φ_kltiₚ ≈ Φ_sink_kltiₚ
    @test Φ_kltiₚ ≈ Φ_src_kltiₚ
end
