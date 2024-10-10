# %%########################################################################################
# DD_local.jl
#
# Compute local DD correlators from perambulators and sparse modes.
#
# Usage:
#   DD_local.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters.
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


# %%###############
# Global Parameters
###################

# Set global parameters
PC.read_parameters()

# Set which momenta should be used
if PC.parms_toml["Momenta"]["p"] == "all"
    p_arr = PC.parms.p_arr
else
    p_arr = PC.parms_toml["Momenta"]["p"]
end

# Get momentum indices in mode doublets corresponding to the momentas in p_arr
iₚ_arr = [findfirst(p_ -> p_ == p, PC.parms.p_arr) for p in p_arr]
if any(isnothing.(iₚ_arr))
    throw(DomainError("a chosen momentum `p` is not contained in the mode doublets."))
end

# Array of (monomial of) γ-matrices and their labels
Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ = length(Γ_arr)
Γ_DD_labels = ["gamma_5", "gamma_1", "gamma_2", "gamma_3", "-i1"]

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(myrank).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end


# %%############
# File Functions
################

# File paths
perambulator_file(n_cnfg, i_src) = PC.parms.perambulator_dir/
    "$(PC.parms_toml["Perambulator"]["label_light"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
perambulator_charm_file(n_cnfg, i_src) = PC.parms.perambulator_charm_dir/
    "$(PC.parms_toml["Perambulator"]["label_charm"])$(i_src)_" *
    "$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"

function write_correlator(n_cnfg, t₀)
    file_path = PC.parms.result_dir/"correlators_DD_local_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    file = HDF5.h5open(string(file_path), "w")

    # Write correlators with dimension labels
    for (iₚ, p) in enumerate(p_arr)
        p_str = "p"*join(p, ",")
        file["Correlators/$p_str"] = C_tnmn̄m̄iₚ[:, :, :, :, :, iₚ]
        HDF5.attrs(file["Correlators/$p_str"])["DIMENSION_LABELS"] = labels
    end

    # Write spin structure
    file["Spin Structure/Gamma_DD_1"] = Γ_DD_labels
    file["Spin Structure/Gamma_DD_2"] = Γ_DD_labels

    # Write parameter file and program information
    file["parms.toml"] = PC.parms.parms_toml_string
    file["Program Information"] = PC.parms_toml["Program Information"]
    
    close(file)

    return
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulators and sparse mode arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

# Correlator and its labels
correlator_size = (PC.parms.Nₜ, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(p_arr))
C_tnmn̄m̄iₚ = Array{ComplexF64}(undef, correlator_size)
# Reversed order in Julia
labels = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "t"]


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    # Compute correlator entries
    @time "      DD_local_contracton!" begin
        PC.DD_local_contractons!(
            C_tnmn̄m̄iₚ, τ_charm_αkβlt, τ_αkβlt, sparse_modes_arrays, Γ_arr, t₀, p_arr)
    end
    println()
end


function main()
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
                
                # Write Correlator
                @time "    Write correlator" begin
                    write_correlator(n_cnfg, t₀)
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

    # Wait until all ranks finished
    MPI.Barrier(comm)

    # Remove finished_cnfgs file
    rm(finished_cnfgs_file, force=true)
end

main()

# %%