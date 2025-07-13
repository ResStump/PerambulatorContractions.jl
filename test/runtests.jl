import MPI
import LinearAlgebra as LA
import TensorOperations as TO
import FilePathsBase: /, Path
import PerambulatorContractions as PC
import Test:@test, @testset


# Run serial tests
include("IO_tests.jl")
include("utils_tests.jl")
include("variables_tests.jl")
include("contractions_tests.jl")

# Run MPI tests
nprocs = clamp(Sys.CPU_THREADS, 3, 4)
@testset "MPI tests" begin
    run(`$(MPI.mpiexec()) -n $nprocs $(Base.julia_cmd()) --project=.. mpi_utils_tests.jl`)
    @test true
end