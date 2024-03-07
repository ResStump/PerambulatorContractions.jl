import Test:@test, @testset

include("../src/IO.jl")
include("../src/utils.jl")

# Add infile manually to arguments
pushfirst!(ARGS, "-i", "test/16x8v1_parameter_files/pseudoscalar_16x8v1.toml")

# Read parameters from infile
parms, parms_toml = read_parameters()

# Paths to files
sparse_modes_file_Nsep1 = parms.sparse_modes_dir/"sparse_modes_Nsep1_16x8v1n1"
sparse_modes_file_Nsep2 = parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"


@testset "Test increase_separation" begin
    sparse_modes_arrays_Nsep1 = read_sparse_modes(sparse_modes_file_Nsep1)
    sparse_modes_arrays_Nsep2 = read_sparse_modes(sparse_modes_file_Nsep2)
    sparse_modes_arrays_new_Nsep2 = increase_separation(sparse_modes_arrays_Nsep1, 2)

    @test sparse_modes_arrays_new_Nsep2 == sparse_modes_arrays_Nsep2
end