import MPI
import Test:@test, @testset


# Run serial tests
include("IO_tests.jl")
include("utils_tests.jl")
include("pseudoscalar_tests.jl")

# Run MPI tests
nprocs = clamp(Sys.CPU_THREADS, 2, 4)
@testset "MPI tests" begin
    run(`$(MPI.mpiexec()) -n $nprocs $(Base.julia_cmd()) test/mpi_utils_tests.jl`)
    @test true
end