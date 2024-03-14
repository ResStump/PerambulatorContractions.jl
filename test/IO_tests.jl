# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

# Paths to files
perambulator_file = PC.parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = PC.parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
sparse_modes_file = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"


@testset "Read files" begin
    # Allocate arrays
    τ_αkβlt = PC.allocate_perambulator()
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file)
    sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file)

    # Read files using '!' Functions
    PC.read_perambulator!(perambulator_file, τ_αkβlt)
    PC.read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)
    PC.read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)

    # Read files using 'return' functions
    τ2_αkβlt = PC.read_perambulator(perambulator_file)
    Φ2_kltiₚ = PC.read_mode_doublets(mode_doublets_file)
    sparse_modes_arrays2 = PC.read_sparse_modes(sparse_modes_file)

    # Compare arrays
    @test τ_αkβlt == τ2_αkβlt
    @test Φ_kltiₚ == Φ2_kltiₚ
    @test sparse_modes_arrays[1] == sparse_modes_arrays2[1]
    @test sparse_modes_arrays[2] == sparse_modes_arrays2[2]
    @test sparse_modes_arrays[3] == sparse_modes_arrays2[3]
    @test sparse_modes_arrays[4] == sparse_modes_arrays2[4]
end
