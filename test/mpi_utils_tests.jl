import MPI
import LinearAlgebra as LA
import Random
import PerambulatorContractions as PC
import Test.@test


MPI.Init()
myrank = MPI.Comm_rank(MPI.COMM_WORLD)
N_ranks = MPI.Comm_size(MPI.COMM_WORLD)

# Test mpi_broadcast
Random.seed!(1234)
f = (x, y, z) -> (LA.tr(x), LA.tr(y.^2), LA.tr(z.^3))
arr1 = [rand(10, 10) for _ in 1:64]
arr2 = [rand(15, 15) for _ in 1:64]
arr3 = [rand(10, 10)]
# Test with rank 0 as root
if myrank == 0
    result = PC.mpi_broadcast(f, arr1, arr2, arr3)
    @test result == f.(arr1, arr2, arr3)
else
    PC.mpi_broadcast(f)
end
# Test with rank 1 as root
if myrank == 1
    result = PC.mpi_broadcast(f, arr1, arr2, arr3, root=1)
    @test result == f.(arr1, arr2, arr3)
else
    PC.mpi_broadcast(f, root=1)
end
