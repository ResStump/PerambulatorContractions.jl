# %%########################################################################################
# dad_local_distributed.jl
#
# Compute local diquark-antidiquark (dad) correlators from perambulators and sparse modes
# where the contractions are done in parallel on each MPI rank using Distributed.jl.
#
# Usage:
#   dad_local_distributed.jl -i <parms file>
#
# where <parms file> is a toml file containing the required parameters.
#
############################################################################################

import Distributed as D

D.@everywhere begin
    import MKL
    import LinearAlgebra as LA
    import MPI
    import HDF5
    import DelimitedFiles as DF
    import FilePathsBase: /, Path
    import BenchmarkTools.@btime
    import PerambulatorContractions as PC
end

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

D.@everywhere begin
    # Broadcast global parameters
    PC.parms = $(PC.parms)
    PC.parms_toml = $(PC.parms_toml)

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
end

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

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulators and sparse mode arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

# Correlator and its labels
correlator_size = (PC.parms.Nₜ, Nᵧ_1, Nᵧ_2, length(p_arr))
C_tnmiₚ = Array{ComplexF64}(undef, correlator_size)
# Reversed order in Julia
labels = ["Gamma2", "Gamma1", "t"]


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    @time "      dad_local_contractons!" begin
        # Index for source time `t₀`
        i_t₀ = t₀+1

        # Unpack sparse modes arrays
        x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

        # Convert arrays to vectors of arrays in the time axis
        τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)
        τ_arr = eachslice(τ_αkβlt, dims=5)
        x_sink_arr = eachslice(x_sink_μiₓt, dims=3)
        v_sink_arr = eachslice(v_sink_ciₓkt, dims=4)

        # Select sink time `t₀`
        x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
        v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]

        # Function to compute contraction
        contraction = (τ_charm, τ, x_sink, v_sink) -> begin
            PC.dad_local_contractons(
                τ_charm, τ, (x_sink, x_src_μiₓ_t₀, v_sink, v_src_ciₓk_t₀),
                Γ₁_arr, Γ₂_arr, p_arr
            )
        end

        # Distribute workload and fetch result
        corr = D.pmap(contraction, τ_charm_arr, τ_arr, x_sink_arr, v_sink_arr)

        # Store correlator entries
        for iₜ in 1:PC.parms.Nₜ
            # Time index for storing correlator entrie
            i_Δt = mod1(iₜ-t₀, PC.parms.Nₜ)
            
            C_tnmiₚ[i_Δt, :, :, :] = corr[iₜ]
        end
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