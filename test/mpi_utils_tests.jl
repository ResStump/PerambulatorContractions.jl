import MPI
import PerambulatorContractions as PC
import Test.@test


MPI.Init()
myrank = MPI.Comm_rank(MPI.COMM_WORLD)
N_ranks = MPI.Comm_size(MPI.COMM_WORLD)

# Add infile manually to arguments
pushfirst!(ARGS, "-i", "16x8v1_parameter_files/mpi_utils_16x8v1.toml")

# Read parameters from infile
PC.read_parameters()

# Allocate momory for correlator
correlator = Array{ComplexF64}(undef, PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg)

for i_cnfg in eachindex(PC.parms.cnfg_indices)
    if PC.is_my_cnfg(i_cnfg)
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
