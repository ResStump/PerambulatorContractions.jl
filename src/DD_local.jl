# %%########################################################################################
# DD_local.jl
#
# Compute DD tetraquark correlators from perambulators, mode doublets and sparse modes
#
# Usage:
#   DD_local.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters
#
############################################################################################

import MKL
import LinearAlgebra as LA
import MPI
import HDF5
import DelimitedFiles as DF
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

# Array of (monomial of) γ-matrices
Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ = length(Γ_arr)


# File paths
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/
    "$(PC.parms_toml["Perambulator"]["label_light"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
perambulator_charm_file(n_cnfg, i_src) = PC.parms.perambulator_charm_dir/
    "$(PC.parms_toml["Perambulator"]["label_charm"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"


# Correlator files
function correlator_file(name, n_cnfg, t₀)
    run_name = PC.parms_toml["Run name"]["name"]

    file = PC.parms.result_dir/"$(run_name)_$(PC.parms.N_modes)modes_$(name)_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    
    return file
end


# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(myrank).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulator and mode doublets arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))

# Sparse mode arrays
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

# Correlator and its labels
correlator_size = (PC.parms.Nₜ, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(PC.parms.p_arr))
C_tnmn̄m̄iₚ = Array{ComplexF64}(undef, correlator_size)
mom_dim = ndims(C_tnmn̄m̄iₚ)
labels = ["t", "Gamma1", "Gamma2", "Gamma1 bar", "Gamma2 bar"]

# Get momentum indices from mode doublets
iₚ_arr = PC.momentum_indices_mode_doublets(mode_doublets_file(n_cnfg))


function compute_contractions!(t₀)
    # Compute correlator entries
    @time "      DD_local_contracton!" begin
        PC.DD_local_contractons!(
            C_tnmn̄m̄iₚ, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ_arr,
            t₀, PC.parms.p_arr)
    end
    println()
end


function main()
    # Computation
    #############

    # Loop over all configurations
    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_indices)
        # Skip the cnfgs this rank doesn't have to compute
        if !PC.is_my_cnfg(i_cnfg)
            continue
        end
        if continuation_run && (n_cnfg in finished_cnfgs)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            @time "  Read mode doublets" begin
                PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
            end
            @time "  Read sparse modes" begin
                PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                @time "    Read perambulators" begin
                    PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                    PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                          τ_charm_αkβlt)
                end
                println()

                compute_contractions!(t₀)
                
                # Store Correlator
                @time "  Write correlator" begin
                    name = "DD_local"
                    PC.write_correlator(correlator_file(name, n_cnfg, t₀), C_tnmn̄m̄iₚ,
                                        PC.parms.p_arr, mom_dim, labels)
                end
                println()
            end

            # Update finished_cnfgs
            push!(finished_cnfgs, n_cnfg)
            DF.writedlm(string(finished_cnfgs_file), finished_cnfgs, '\n')
        end
        println("\n")

        # Run garbage collector
        GC.gc()
    end

    # Remove finished_cnfgs file
    rm(finished_cnfgs_file, force=true)
end

main()

# %%