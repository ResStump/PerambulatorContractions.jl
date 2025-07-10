# %%########################################################################################
# DD_mixed.jl
#
# Compute mixed local-nonlocal DD correlators from perambulators, mode doublets and
# sparse modes where the contractions are done in parallel using MPI.jl.
#
# Usage:
#   DD_mixed.jl -i <parms file> --nranks-per-cnfg <n>
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

# Array of (monomial of) γ-matrices and their labels
Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ = length(Γ_arr)
Γ_DD_labels = ["gamma_5", "gamma_1", "gamma_2", "gamma_3", "i1"]

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(comm_number).txt"
continuation_run = PC.parms_toml["Various"]["continuation_run"]
if continuation_run
    finished_cnfgs = vec(DF.readdlm(string(finished_cnfgs_file), '\n', Int))
else
    finished_cnfgs = []
end

# Shape of correlator
direction = PC.parms_toml["Correlator shape"]["direction"]
nₜ = PC.parms_toml["Correlator shape"]["n_timeslices"]
if direction != "full" && 2*nₜ > PC.parms.Nₜ
    throw(ArgumentError("Invalid number of timeslices: $nₜ"))
end


# %%###################################
# Momentum Pairs for Nonlocal Operators
#######################################

# Array of square of total angular momentas
Ptot_sq_arr = PC.parms_toml["Momenta nonlocal"]["Ptot_sq"]

# Maximal sum of squares of the momentum pairs that are used
p_sq_sum_max_arr = PC.parms_toml["Momenta nonlocal"]["p_sq_sum_max"]

# Compute all (relevant) momentum index pairs
Iₚ_nonlocal_arr = []
for (Ptot_sq, p_sq_sum_max) in zip(Ptot_sq_arr, p_sq_sum_max_arr)
    Iₚ_arr = PC.generate_momentum_pairs(Ptot_sq, p_sq_sum_max)
    append!(Iₚ_nonlocal_arr, Iₚ_arr)
end

# Divide up momenta into chunks
mom_chunk_size = PC.parms_toml["Momenta nonlocal"]["chunk_size"]
if mom_chunk_size == "full"
    global mom_chunk_size = length(Iₚ_nonlocal_arr)
end
Iₚ_nonlocal_chunk_arr = collect(Iterators.partition(Iₚ_nonlocal_arr, mom_chunk_size))


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
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"

function write_correlator(n_cnfg, t₀, mom_chunk_idx, mode="w")
    if mode ∉ ["r+", "w"]
        throw(ArgumentError("Invalid mode: $mode, must be 'r+' or 'w'"))
    end

    file_path = PC.parms.result_dir/"correlators_DD_mixed_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    file = HDF5.h5open(string(file_path), mode)

    # Loop over all momentum index pairs for the nonlocal operator in current chunk
    for (iₚ_nonlocal, Iₚ_nonlocal) in enumerate(Iₚ_nonlocal_chunk_arr[mom_chunk_idx])
        # Get momenta
        p₁, p₂ = PC.parms.p_arr[Iₚ_nonlocal]
        Ptot = p₁ + p₂

        # Paths to groups in hdf5 file
        Ptot_str = join(Ptot, ",")
        p₁_str = join(p₁, ",")
        group_nloc_loc =
            "Correlators/Ptot$(Ptot_str)/p_nonlocal1_$(p₁_str)/nonlocal-local"
        group_loc_nloc =
            "Correlators/Ptot$(Ptot_str)/p_nonlocal1_$(p₁_str)/local-nonlocal"

        # Write correlators with dimension labels
        if direction == "full"
            # Remove forward/backward dimension
            file[group_nloc_loc] = 
                C_nonlocal_local_tdnmn̄m̄iₚIₚ[:, 1, :, :, :, :, 1, iₚ_nonlocal]
            file[group_loc_nloc] = 
                C_local_nonlocal_tdnmn̄m̄iₚIₚ[:, 1, :, :, :, :, 1, iₚ_nonlocal]
        else
            file[group_nloc_loc] = 
                C_nonlocal_local_tdnmn̄m̄iₚIₚ[:, :, :, :, :, :, 1, iₚ_nonlocal]
            file[group_loc_nloc] = 
                C_local_nonlocal_tdnmn̄m̄iₚIₚ[:, :, :, :, :, :, 1, iₚ_nonlocal]
        end
        HDF5.attrs(file[group_nloc_loc])["DIMENSION_LABELS"] = labels
        HDF5.attrs(file[group_loc_nloc])["DIMENSION_LABELS"] = labels
    end

    if mode == "w"
        # Write correlator shape
        file["Correlator shape"] = direction

        # Write spin structure
        file["Spin Structure/Gamma_DD_1"] = Γ_DD_labels
        file["Spin Structure/Gamma_DD_2"] = Γ_DD_labels

        # Write parameter file and program information
        file["parms.toml"] = PC.parms.parms_toml_string
        file["Program Information"] = PC.parms_toml["Program Information"]
    end

    close(file)

    return
end


# %%#############
# Allocate Arrays
#################

if my_cnfg_rank == 0
    # Select valid cnfg number
    n_cnfg = PC.parms.cnfg_numbers[1]

    # Perambulators, mode doublets and sparse modes arrays
    τ_αkβlt = PC.allocate_perambulator()
    τ_charm_αkβlt = PC.allocate_perambulator()
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))
    sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

    # Correlator and its labels (order of labels reversed in Julia)
    # (for momentum of local operator always choose Ptot -> index always iₚ=1)
    if direction in ["forward", "backward"]
        correlator_size = (nₜ, 1, Nᵧ, Nᵧ, Nᵧ, Nᵧ, 1, mom_chunk_size)
    elseif direction == "forward/backward"
        correlator_size = (nₜ, 2, Nᵧ, Nᵧ, Nᵧ, Nᵧ, 1, mom_chunk_size)
    elseif direction == "full"
        correlator_size = (PC.parms.Nₜ, 1, Nᵧ, Nᵧ, Nᵧ, Nᵧ, 1, mom_chunk_size)
    else
        throw(ArgumentError("Invalid direction: $direction"))
    end
    C_nonlocal_local_tdnmn̄m̄iₚIₚ = Array{ComplexF64}(undef, correlator_size)
    C_local_nonlocal_tdnmn̄m̄iₚIₚ = Array{ComplexF64}(undef, correlator_size)
    if direction == "full"
        labels = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "t"]
    else
        labels = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "fwd/bwd", "t"]
    end
end


# %%#########
# Computation
#############

function compute_contractions!(t₀, mom_chunk_idx)
    @time "      DD mixed contractons" begin
        if my_cnfg_rank == 0
            # Index for source time `t₀`
            i_t₀ = t₀+1

            # Time range to be computed
            if direction == "forward"
                iₜ_range = i_t₀ .+ (0:nₜ-1)
            elseif direction == "backward"
                iₜ_range = i_t₀ .+ (1-nₜ:0)
            elseif direction == "forward/backward"
                iₜ_range = i_t₀ .+ (1-nₜ:nₜ-1)
            else
                iₜ_range = i_t₀ .+ (0:PC.parms.Nₜ-1)
            end
            iₜ_range = mod1.(iₜ_range, PC.parms.Nₜ)

            # Unpack sparse modes arrays
            x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

            # Convert arrays to vectors of arrays in the time axis
            τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)[iₜ_range]
            τ_arr = eachslice(τ_αkβlt, dims=5)[iₜ_range]
            Φ_arr = eachslice(Φ_kltiₚ, dims=3)[iₜ_range]
            x_sink_arr = eachslice(x_sink_μiₓt, dims=3)[iₜ_range]
            v_sink_arr = eachslice(v_sink_ciₓkt, dims=4)[iₜ_range]

            # Select source time `t₀`
            Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]
            x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
            v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]
        end

        # Function to compute contractions
        contractions = (τ_charm, τ, Φ, Φ_t₀, x_sink, x_src, v_sink, v_src) -> begin
            C_nl_arr = []
            C_ln_arr = []

            for Iₚ_nonlocal in Iₚ_nonlocal_chunk_arr[mom_chunk_idx]
                p₁, p₂ = PC.parms.p_arr[Iₚ_nonlocal]
                Ptot = p₁ + p₂

                # Contraction for correlator of form 
                # <O_nonlocal O_local^†> and <O_local O_nonlocal^†>
                C_nl, C_ln = PC.DD_mixed_contractons(
                    τ_charm, τ, Φ, Φ_t₀, (x_sink, x_src, v_sink, v_src), Γ_arr,
                    Iₚ_nonlocal, [Ptot]
                )
                push!(C_nl_arr, C_nl)
                push!(C_ln_arr, C_ln)
            end
            
            # Run garbage collector for "young" objects
            GC.gc(false)

            # Return as contiguous arrays
            return stack(C_nl_arr), stack(C_ln_arr)
        end

        # Distribute workload and compute contractions
        if my_cnfg_rank == 0
            corr_arr = PC.mpi_broadcast(
                contractions, τ_charm_arr, τ_arr, Φ_arr, [Φ_kliₚ_t₀],
                x_sink_arr, [x_src_μiₓ_t₀], v_sink_arr, [v_src_ciₓk_t₀], comm=cnfg_comm,
                log_prefix="      "
            )
        else
            PC.mpi_broadcast(contractions, comm=cnfg_comm)
        end

        # Store correlator entries
        if my_cnfg_rank == 0
            # Number of Momentum pairs
            Iₚ_len = length(Iₚ_nonlocal_chunk_arr[mom_chunk_idx])

            if direction == "forward/backward"
                # Backward direction
                for iₜ in 1:nₜ-1
                    C_nonlocal_local_tdnmn̄m̄iₚIₚ[iₜ, 2, :, :, :, :, :, 1:Iₚ_len] =
                        corr_arr[iₜ][1]
                    C_local_nonlocal_tdnmn̄m̄iₚIₚ[iₜ, 2, :, :, :, :, :, 1:Iₚ_len] =
                        corr_arr[iₜ][2]
                end

                # Source time
                C_nonlocal_local_tdnmn̄m̄iₚIₚ[nₜ, 2, :, :, :, :, :, 1:Iₚ_len] =
                    corr_arr[nₜ][1]
                C_local_nonlocal_tdnmn̄m̄iₚIₚ[nₜ, 2, :, :, :, :, :, 1:Iₚ_len] =
                    corr_arr[nₜ][2]
                C_nonlocal_local_tdnmn̄m̄iₚIₚ[1, 1, :, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][1]
                C_local_nonlocal_tdnmn̄m̄iₚIₚ[1, 1, :, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][2]

                # Forward direction
                for iₜ in 2:nₜ
                    C_nonlocal_local_tdnmn̄m̄iₚIₚ[iₜ, 1, :, :, :, :, :, 1:Iₚ_len] =
                        corr_arr[iₜ+nₜ-1][1]
                    C_local_nonlocal_tdnmn̄m̄iₚIₚ[iₜ, 1, :, :, :, :, :, 1:Iₚ_len] = 
                        corr_arr[iₜ+nₜ-1][2]
                end
            else
                for (iₜ, corr_t) in enumerate(corr_arr)    
                    C_nonlocal_local_tdnmn̄m̄iₚIₚ[iₜ, 1, :, :, :, :, :, 1:Iₚ_len] = corr_t[1]
                    C_local_nonlocal_tdnmn̄m̄iₚIₚ[iₜ, 1, :, :, :, :, :, 1:Iₚ_len] = corr_t[2]
                end
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
                @time "  Read mode doublets" begin
                    PC.read_mode_doublets!(mode_doublets_file(n_cnfg), Φ_kltiₚ)
                end
                @time "  Read sparse modes" begin
                    PC.read_sparse_modes!(sparse_modes_file(n_cnfg), sparse_modes_arrays)
                end
            end
            println()

            # Loop over all sources
            for (i_src, t₀) in enumerate(PC.parms.tsrc_arr[i_cnfg, :])
                println("  Source: $i_src of $(PC.parms.N_src)")

                if my_cnfg_rank == 0
                    @time "    Read perambulators" begin
                        PC.read_perambulator!(perambulator_file(n_cnfg, t₀), τ_αkβlt)
                        PC.read_perambulator!(perambulator_charm_file(n_cnfg, t₀),
                                              τ_charm_αkβlt)
                    end
                end
                println()

                for mom_chunk_idx in eachindex(Iₚ_nonlocal_chunk_arr)
                    println("    Momenta chunk: " *
                            "$mom_chunk_idx of $(length(Iₚ_nonlocal_chunk_arr))")

                    compute_contractions!(t₀, mom_chunk_idx)
                    
                    # Write Correlator
                    if my_cnfg_rank == 0
                        @time "      Write correlator" begin
                            if mom_chunk_idx == 1
                                write_correlator(n_cnfg, t₀, mom_chunk_idx, "w")
                            else
                                write_correlator(n_cnfg, t₀, mom_chunk_idx, "r+")
                            end
                        end
                    end
                    println()

                    # Run garbage collector (fully)
                    GC.gc()
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
