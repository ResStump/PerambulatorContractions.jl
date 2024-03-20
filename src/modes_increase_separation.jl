# %%########################################################################################
# modes_increase_separation.jl
#
# Read the sparse modes increase the separation in the sparce space and write them to disk
#
# Usage:
#   modes_increase_separation.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
############################################################################################

import MKL
import LinearAlgebra as LA
import MPI
import HDF5
import FilePathsBase: /, Path
import BenchmarkTools.@btime
import PerambulatorContractions as PC
#= include("PerambulatorContractions.jl")
PC = PerambulatorContractions =#

# Add infile manually to arguments
# pushfirst!(ARGS, "-i", "run_pseudoscalar/input/pseudoscalar_16x8v1.toml")

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)
N_ranks = MPI.Comm_size(comm)

if myrank != 0
    redirect_stdout(devnull)
end


# %%#############################
# Global Parameters and Functions
#################################

# Set global parameters
PC.read_parameters()


# File paths
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file_new(n_cnfg) = 
    Path(PC.parms_toml["Directories and Files"]["sparse_modes_dir_new"])/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Sparse mode arrays
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

N_sep_new = PC.parms_toml["Increase Separation"]["N_sep_new"]
N_points = prod(PC.parms.Nₖ)÷N_sep_new^3 
sparse_modes_arrays_new = PC.allocate_sparse_modes(N_points=N_points)



function main()
    # Computation
    #############

    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_indices)
        # Skip the cnfgs this rank doesn't have to compute
        if !PC.is_my_cnfg(i_cnfg)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            @time "  Read sparse modes " begin
                PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
            end

            @time "  Increase separation" begin
                PC.increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays,
                                        N_sep_new)
            end

            @time "  Write new sparse modes" begin
                PC.write_sparse_modes(sparse_modes_file_new(n_cnfg),
                                      sparse_modes_arrays_new)
            end
            println()
        end
        println("\n")
    end
end

main()


# %%