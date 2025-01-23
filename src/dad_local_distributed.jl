# %%########################################################################################
# dad_local_distributed.jl
#
# Compute local diquark-antidiquark (dad) correlators from perambulators and sparse modes
# where the contractions are done in parallel using MPI.jl.
#
# Usage:
#   dad_local_distributed.jl -i <parms file> --nranks-per-cnfg <n>
#
# where <parms file> is a toml file containing the required parameters and <n> is the number
# of ranks that simultaneously work on one configuration. If --nranks-per-cnfg is not
# provided, the default value is 1.
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

# Split communicators
cnfg_comm, comm_number, my_cnfgs = PC.cnfg_comm()
my_cnfg_rank = MPI.Comm_rank(cnfg_comm)

# Set which momenta should be used
if PC.parms_toml["Momenta"]["p"] == "all"
    p_arr = PC.parms.p_arr
else
    p_arr = PC.parms_toml["Momenta"]["p"]
end

# Array of (monomial of) γ-matrices
Γ₁_arr = [PC.γ[1], PC.γ[2], PC.γ[3]]
Γ₂_arr = [PC.γ[5]]
Nᵧ_1 = length(Γ₁_arr)
Nᵧ_2 = length(Γ₂_arr)
Γ₁_dad_labels = ["Cgamma_1", "Cgamma_2", "Cgamma_3"]
Γ₂_dad_labels = ["Cgamma_5"]

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(comm_number).txt"
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
    file_path = PC.parms.result_dir/"correlators_dad_local_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    file = HDF5.h5open(string(file_path), "w")

    # Write correlators with dimension labels
    for (iₚ, p) in enumerate(p_arr)
        p_str = "p"*join(p, ",")
        file["Correlators/$p_str"] = C_tnmiₚ[:, :, :, iₚ]
        HDF5.attrs(file["Correlators/$p_str"])["DIMENSION_LABELS"] = labels
    end

    # Write spin structure
    file["Spin Structure/Gamma_dad_1"] = Γ₁_dad_labels
    file["Spin Structure/Gamma_dad_2"] = Γ₂_dad_labels

    # Write parameter file and program information
    file["parms.toml"] = PC.parms.parms_toml_string
    file["Program Information"] = PC.parms_toml["Program Information"]
    
    close(file)

    return
end


# %%#############
# Allocate Arrays
#################

if my_cnfg_rank == 0
    # Select valid cnfg number
    n_cnfg = PC.parms.cnfg_numbers[1]

    # Perambulators and sparse mode arrays
    τ_αkβlt = PC.allocate_perambulator()
    τ_charm_αkβlt = PC.allocate_perambulator()
    sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

    # Correlator and its labels
    correlator_size = (PC.parms.Nₜ, Nᵧ_1, Nᵧ_2, length(p_arr))
    C_tnmiₚ = Array{ComplexF64}(undef, correlator_size)
    # Reversed order in Julia
    labels = ["Gamma2", "Gamma1", "t"]
end


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    @time "      dad_local_contractons" begin
        if my_cnfg_rank == 0
            # Index for source time `t₀`
            i_t₀ = t₀+1

            # Unpack sparse modes arrays
            x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

            # Convert arrays to vectors of arrays in the time axis
            τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)
            τ_arr = eachslice(τ_αkβlt, dims=5)
            x_sink_arr = eachslice(x_sink_μiₓt, dims=3)
            v_sink_arr = eachslice(v_sink_ciₓkt, dims=4)

            # Select source time `t₀`
            x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
            v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]
        end

        # Function to compute contraction
        contraction = (τ_charm, τ, x_sink, x_src, v_sink, v_src) -> begin
            PC.dad_local_contractons(
                τ_charm, τ, (x_sink, x_src, v_sink, v_src),
                Γ₁_arr, Γ₂_arr, p_arr
            )
        end

        # Distribute workload and compute contraction
        if my_cnfg_rank == 0
            corr = PC.mpi_broadcast(contraction, τ_charm_arr, τ_arr, x_sink_arr,
                                    [x_src_μiₓ_t₀], v_sink_arr, [v_src_ciₓk_t₀],
                                    comm=cnfg_comm)
        else
            PC.mpi_broadcast(contraction, comm=cnfg_comm)
        end

        # Store correlator entries
        if my_cnfg_rank == 0
            for iₜ in 1:PC.parms.Nₜ
                # Time index for storing correlator entry
                i_Δt = mod1(iₜ-t₀, PC.parms.Nₜ)
                
                C_tnmiₚ[i_Δt, :, :, :] = corr[iₜ]
            end
        end
    end

    println()
end


function main()
    # Loop over all configurations
    for (i_cnfg, n_cnfg) in enumerate(PC.parms.cnfg_numbers)
        # Skip the cnfgs this rank doesn't have to compute
        if n_cnfg ∉ my_cnfgs
            continue
        end
        if continuation_run && (n_cnfg in finished_cnfgs)
            continue
        end

        println("Configuration $n_cnfg")
        @time "Finished configuration $n_cnfg" begin
            if my_cnfg_rank == 0
                @time "  Read sparse modes" begin
                    PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
                end
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                @time "    Read perambulators" begin
                    if my_cnfg_rank == 0
                        PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                        PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                              τ_charm_αkβlt)
                    end
                end
                println()

                compute_contractions!(t₀)
                
                # Write Correlator
                if my_cnfg_rank == 0
                    @time "    Write correlator" begin
                        write_correlator(n_cnfg, t₀)
                    end
                end
                println()
            end

            # Update finished_cnfgs
            push!(finished_cnfgs, n_cnfg)
            if my_cnfg_rank == 0
                DF.writedlm(string(finished_cnfgs_file), finished_cnfgs, '\n')
            end
        end
        println("\n")

        # Run garbage collector
        GC.gc()
    end

    # Wait until all ranks finished
    MPI.Barrier(comm)

    # Remove finished_cnfgs file
    if my_cnfg_rank == 0
        rm(finished_cnfgs_file, force=true)
    end

    println("Program finished successfully.")
end

main()

# %%