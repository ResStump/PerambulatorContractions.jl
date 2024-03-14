# %%########################################################################################
# pseudoscalar.jl
#
# Compute the pseudoscalar meson from perambulators and mode doublets
#
# Usage:
#   pseudoscalar.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
############################################################################################

import MKL
import LinearAlgebra as LA
import TensorOperations as TO
import MPI
import TOML
import HDF5
import DelimitedFiles
import FilePathsBase: /, Path
import BenchmarkTools.@btime
import PerambulatorContractions as PC

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
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/"perambulator_" *
    "$(PC.parms_toml["Perambulator"]["label_base"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"


# Correlator file names
run_name = PC.parms_toml["Run name"]["name"]
    
correlator_file_tmp = 
    "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar_tmp$(myrank).hdf5"
correlator2_file_tmp = 
    "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar_p0_tmp$(myrank).hdf5"
correlator3_file_tmp = 
    "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar_sparse_tmp$(myrank).hdf5"


correlator_file = "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar.hdf5"
correlator2_file = "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar_p0.hdf5"
correlator3_file = "$(run_name)_$(PC.parms.N_modes)modes_pseudoscalar_sparse.hdf5"


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulator and mode doublets arrays
τ_αkβlt = PC.allocate_perambulator()
Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))

# Sparse mode arrays
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))
if PC.parms_toml["Increased Separation"]["increase_sep"]
    N_sep_new = PC.parms_toml["Increased Separation"]["N_sep_new"]
    N_points = prod(PC.parms.Nₖ)÷N_sep_new^3 
    sparse_modes_arrays_new = PC.allocate_sparse_modes(N_points=N_points)
end

correlator = Array{ComplexF64}(undef, PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg)
correlator2 = Array{ComplexF64}(undef, PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg)
correlator3 = Array{ComplexF64}(undef, PC.parms.Nₜ, PC.parms.N_src, PC.parms.N_cnfg)




function main()
    # Get momentum index
    p_arr = PC.read_mode_doublet_momenta(mode_doublets_file(PC.parms.cnfg_indices[1]))
    iₚ = findfirst(p -> p == PC.parms.p, eachrow(p_arr))
    if isnothing(iₚ)
        throw(DomainError("the chosen momentum 'p' is not contained in mode doublets."))
    end

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
                if PC.parms_toml["Increased Separation"]["increase_sep"]
                    local N_sep_new = PC.parms_toml["Increased Separation"]["N_sep_new"]
                    PC.increase_separation!(sparse_modes_arrays_new, sparse_modes_arrays,
                                         N_sep_new)
                end
            end
            @time "  Read mode doublets" begin
                PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
            end
            println()

            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                @time "    Read perambulator" begin
                    PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                end
                println()

                Cₜ = @view correlator[:, i_src, i_cnfg]
                Cₜ_2 = @view correlator2[:, i_src, i_cnfg]
                Cₜ_3 = @view correlator3[:, i_src, i_cnfg]
                @time "    pseudoscalar_contraction!       " begin
                    PC.pseudoscalar_contraction!(Cₜ, τ_αkβlt, Φ_kltiₚ, t₀, iₚ)
                end
                @time "    pseudoscalar_contraction_p0!    " begin
                    PC.pseudoscalar_contraction_p0!(Cₜ_2, τ_αkβlt, t₀)
                end
                @time "    pseudoscalar_sparse_contraction!" begin
                    if PC.parms_toml["Increased Separation"]["increase_sep"]
                        PC.pseudoscalar_sparse_contraction!(
                            Cₜ_3, τ_αkβlt, sparse_modes_arrays_new, t₀, PC.parms.p
                        )
                    else
                        PC.pseudoscalar_sparse_contraction!(
                            Cₜ_3, τ_αkβlt, sparse_modes_arrays, t₀, PC.parms.p
                        )
                    end
                end
                println()
            end

            # Temporary store correlators 
            @time "  Write tmp correlators" begin
                PC.write_correlator(PC.parms.result_dir/correlator_file_tmp, correlator)
                PC.write_correlator(PC.parms.result_dir/correlator2_file_tmp, correlator2,
                                    zeros(Int, 3))
                PC.write_correlator(PC.parms.result_dir/correlator3_file_tmp, correlator3)
            end
            println()
        end
        println("\n")
    end

    # Broadcast correlator to all ranks
    @time "Broadcast correlators" begin
        PC.broadcast_correlators!(correlator)
        PC.broadcast_correlators!(correlator2)
        PC.broadcast_correlators!(correlator3)
    end

    # Store correlator and remove tmp correlators
    #############################################

    if myrank == 0
        @time "Write correlators" begin
            PC.write_correlator(PC.parms.result_dir/correlator_file, correlator)
            PC.write_correlator(PC.parms.result_dir/correlator2_file, correlator2,
                                zeros(Int, 3))
            PC.write_correlator(PC.parms.result_dir/correlator3_file, correlator3)
        end
    end

    @time "Remove tmp correlators" begin
        rm(PC.parms.result_dir/correlator_file_tmp, force=true)
        rm(PC.parms.result_dir/correlator2_file_tmp, force=true)
        rm(PC.parms.result_dir/correlator3_file_tmp, force=true)
    end

end

main()


# %%