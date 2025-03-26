# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/16x8v1.toml")

# Read parameters from infile
PC.read_parameters(:meson)

# Paths to files
perambulator_file = PC.parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
mode_triplets_file = PC.parms.mode_triplets_dir/"mode_triplets_16x8v1n1"
sparse_modes_file = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"
sparse_modes_file_tmp = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1_tmp"

# sparse_modes_arrays
sparse_modes_arrays = nothing
sparse_modes_arrays2 = nothing

@testset "Read files" begin
    # Allocate arrays
    τ_αkβlt = PC.allocate_perambulator()
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file)
    Φ_Ktiₚ = PC.allocate_mode_triplets(mode_triplets_file)
    global sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file)

    # Read files using '!' Functions
    PC.read_perambulator!(perambulator_file, τ_αkβlt)
    PC.read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)
    PC.read_mode_triplets!(mode_triplets_file, Φ_Ktiₚ)
    PC.read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)

    # Read files using 'return' functions
    τ2_αkβlt = PC.read_perambulator(perambulator_file)
    Φ2_kltiₚ = PC.read_mode_doublets(mode_doublets_file)
    Φ2_Ktiₚ = PC.read_mode_triplets(mode_triplets_file)
    global sparse_modes_arrays2 = PC.read_sparse_modes(sparse_modes_file)

    # Compare arrays
    @test τ_αkβlt == τ2_αkβlt
    @test Φ_kltiₚ == Φ2_kltiₚ
    @test Φ_Ktiₚ == Φ2_Ktiₚ
    @test sparse_modes_arrays == sparse_modes_arrays2
end

@testset "Write files" begin
    # Write and read sparse_modes_arrays
    PC.write_sparse_modes(sparse_modes_file_tmp, sparse_modes_arrays)
    PC.read_sparse_modes!(sparse_modes_file_tmp, sparse_modes_arrays2)

    # Compare written and read arrays
    @test sparse_modes_arrays == sparse_modes_arrays2

    rm(sparse_modes_file_tmp)
end
