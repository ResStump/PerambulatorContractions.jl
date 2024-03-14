# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

# Paths to files
sparse_modes_file_Nsep1 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"
sparse_modes_file_Nsep2 = PC.parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"


@testset "Test increase_separation" begin
    sparse_modes_arrays_Nsep1 = PC.read_sparse_modes(sparse_modes_file_Nsep1)
    sparse_modes_arrays_Nsep2 = PC.read_sparse_modes(sparse_modes_file_Nsep2)
    sparse_modes_arrays_new_Nsep2 = PC.increase_separation(sparse_modes_arrays_Nsep1, 2)

    @test sparse_modes_arrays_new_Nsep2 == sparse_modes_arrays_Nsep2
end