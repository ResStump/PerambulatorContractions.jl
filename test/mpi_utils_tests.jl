import MPI
import LinearAlgebra as LA
import Random
import PerambulatorContractions as PC
import Test.@test


MPI.Init()
myrank = MPI.Comm_rank(MPI.COMM_WORLD)
N_ranks = MPI.Comm_size(MPI.COMM_WORLD)

# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/mpi_utils_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

# Split communicators
cnfg_comm, comm_number, my_cnfgs = PC.cnfg_comm()
my_cnfg_rank = MPI.Comm_rank(cnfg_comm)

# Allocate memory for correlator
correlator = Array{ComplexF64}(undef, PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg)

for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_numbers)
    if n_cnfg in my_cnfgs
        correlator[:, :, i_cnfg] .= myrank
    end
end

# Make copy for sending
correlator2 = copy(correlator)

# Broadcast and send
PC.broadcast_correlators!(correlator)
PC.send_correlator_to_root!(correlator2)

# Check if communication worked
N_cnfg_per_rank = ceil(Int, PC.parms.N_cnfg/N_ranks)

rank_cnfg_indices = @. ((1:PC.parms.N_cnfg) - 1) ÷ N_cnfg_per_rank
rank_cnfg_indices = reshape(rank_cnfg_indices, 1, 1, :)

# Test broadcasted correlator on all ranks
@test all(correlator .== rank_cnfg_indices)

# Only test sent correlator on rank 0
if myrank == 0
    @test all(correlator2 .== rank_cnfg_indices)
end


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
