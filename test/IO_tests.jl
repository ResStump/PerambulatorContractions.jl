include("../src/allocate_arrays.jl")

# Paths to files
perambulator_file = parms.perambulator_dir/"perambulator_light_tsrc2_16x8v1n1"
mode_doublets_file = parms.mode_doublets_dir/"mode_doublets_16x8v1n1"
sparse_modes_file = parms.sparse_modes_dir/"sparse_modes_Nsep2_16x8v1n1"


@testset "Read files" begin
    # Allocate arrays
    τ_αkβlt = allocate_perambulator()
    Φ_kltiₚ = allocate_mode_doublets(mode_doublets_file)
    sparse_modes_arrays = allocate_sparse_modes(sparse_modes_file)

    # Read files using '!' Functions
    read_perambulator!(perambulator_file, τ_αkβlt)
    read_mode_doublets!(mode_doublets_file, Φ_kltiₚ)
    read_sparse_modes!(sparse_modes_file, sparse_modes_arrays)

    # Read files using 'return' functions
    τ2_αkβlt = read_perambulator(perambulator_file)
    Φ2_kltiₚ = read_mode_doublets(mode_doublets_file)
    sparse_modes_arrays2 = read_sparse_modes(sparse_modes_file)

    # Compare arrays
    @test τ_αkβlt == τ2_αkβlt
    @test Φ_kltiₚ == Φ2_kltiₚ
    @test sparse_modes_arrays[1] == sparse_modes_arrays2[1]
    @test sparse_modes_arrays[2] == sparse_modes_arrays2[2]
    @test sparse_modes_arrays[3] == sparse_modes_arrays2[3]
    @test sparse_modes_arrays[4] == sparse_modes_arrays2[4]
end
