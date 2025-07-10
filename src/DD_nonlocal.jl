# %%########################################################################################
# DD_nolocal.jl
#
# Compute nonlocal DD correlators from perambulators and mode doublets where the
# contractions are done in parallel using MPI.jl.
#
# Usage:
#   DD_nolocal.jl -i <parms file> --nranks-per-cnfg <n>
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


# %%############################
# Generate Momentum Combinations
################################

# Array of square of total angular momentas
Ptot_sq_arr = PC.parms_toml["Momenta"]["Ptot_sq"]

# Maximal sum of squares of the momentum pairs that are used
p_sq_sum_max_arr = PC.parms_toml["Momenta"]["p_sq_sum_max"]

# Compute all combinations of the momentum indeces at sink and source
Iₚ_arr = []
for (Ptot_sq, p_sq_sum_max) in zip(Ptot_sq_arr, p_sq_sum_max_arr)
    # 4-tuple of momenta
    Iₚ_arr_ = PC.generate_momentum_4tuples(Ptot_sq, p_sq_sum_max)
    append!(Iₚ_arr, Iₚ_arr_)
end

# Divide up momenta into chunks
mom_chunk_size = PC.parms_toml["Momenta"]["chunk_size"]
if mom_chunk_size == "full"
    global mom_chunk_size = length(Iₚ_arr)
end
Iₚ_chunk_arr = collect(Iterators.partition(Iₚ_arr, mom_chunk_size))


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

function write_correlator(n_cnfg, t₀, mom_chunk_idx, mode="w")
    if mode ∉ ["r+", "w"]
        throw(ArgumentError("Invalid mode: $mode, must be 'r+' or 'w'"))
    end

    file_path = PC.parms.result_dir/"correlators_DD_nonlocal_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    file = HDF5.h5open(string(file_path), mode)

    # Loop over all momentum index combinations in current chunk
    for (i_p, Iₚ) in enumerate(Iₚ_chunk_arr[mom_chunk_idx])
        # Get momenta
        p₁, p₂, p₃, p₄ = PC.parms.p_arr[Iₚ]
        @assert p₁ + p₂ == p₃ + p₄

        # Paths to groups in hdf5 file
        Ptot_str = join(p₁ + p₂, ",")
        p₁_str, p₃_str = join.([p₁, p₃], ",")
        group_ūcd̄c_c̄uc̄d = "Correlators/Ptot$(Ptot_str)/psink1_$(p₁_str)/" *
            "psrc1_$(p₃_str)/ubar_c_dbar_c-cbar_u_cbar_d"
        group_ūcd̄c_c̄dc̄u = "Correlators/Ptot$(Ptot_str)/psink1_$(p₁_str)/" *
            "psrc1_$(p₃_str)/ubar_c_dbar_c-cbar_d_cbar_u"

        # Write correlators with dimension labels
        if direction == "full"
            # Remove forward/backward dimension
            file[group_ūcd̄c_c̄uc̄d] = 
                C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[:, 1, :, :, :, :, i_p]
            file[group_ūcd̄c_c̄dc̄u] = 
                C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[:, 1, :, :, :, :, i_p]
        else
            mom_dim = ndims(C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ)
            file[group_ūcd̄c_c̄uc̄d] = selectdim(C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ, mom_dim, i_p)
            file[group_ūcd̄c_c̄dc̄u] = selectdim(C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ, mom_dim, i_p)
        end
        HDF5.attrs(file[group_ūcd̄c_c̄uc̄d])["DIMENSION_LABELS"] = labels
        HDF5.attrs(file[group_ūcd̄c_c̄dc̄u])["DIMENSION_LABELS"] = labels
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

    # Perambulators and mode doublets arrays
    τ_αkβlt = PC.allocate_perambulator()
    τ_charm_αkβlt = PC.allocate_perambulator()
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))

    # Correlator and its labels (order of labels reversed in Julia)
    if direction in ["forward", "backward"]
        correlator_size = (nₜ, 1, Nᵧ, Nᵧ, Nᵧ, Nᵧ, mom_chunk_size)
    elseif direction == "forward/backward"
        correlator_size = (nₜ, 2, Nᵧ, Nᵧ, Nᵧ, Nᵧ, mom_chunk_size)
    elseif direction == "full"
        correlator_size = (PC.parms.Nₜ, 1, Nᵧ, Nᵧ, Nᵧ, Nᵧ, mom_chunk_size)
    else
        throw(ArgumentError("Invalid direction: $direction"))
    end
    C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ = Array{ComplexF64}(undef, correlator_size)
    C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ = Array{ComplexF64}(undef, correlator_size)
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
    @time "      DD nolocal contractons" begin
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

            # Convert arrays to vectors of arrays in the time axis
            τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)[iₜ_range]
            τ_arr = eachslice(τ_αkβlt, dims=5)[iₜ_range]
            Φ_arr = eachslice(Φ_kltiₚ, dims=3)[iₜ_range]

            # Select source time `t₀`
            Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]
        end

        # Function to compute contractions
        contractions = (τ_charm, τ, Φ_t, Φ_t₀) -> begin
            C1_arr = []
            C2_arr = []

            # Loop over all momentum index combinations in current chunk
            for Iₚ in Iₚ_chunk_arr[mom_chunk_idx]
                C1 = PC.DD_nonlocal_contractons(τ_charm, τ, Φ_t, Φ_t₀, Γ_arr, Iₚ)
                C2 = PC.DD_nonlocal_contractons(τ_charm, τ, Φ_t, Φ_t₀, Γ_arr, Iₚ,
                                                swap_ud=true)
                push!(C1_arr, C1)
                push!(C2_arr, C2)
            end
            
            # Run garbage collector for "young" objects
            GC.gc(false)

            # Return as contiguous arrays
            return stack(C1_arr), stack(C2_arr)
        end

        # Distribute workload and compute contraction
        if my_cnfg_rank == 0
            corr_arr = PC.mpi_broadcast(contractions, τ_charm_arr, τ_arr, Φ_arr,
                                        [Φ_kliₚ_t₀], comm=cnfg_comm, log_prefix="      ")
        else
            PC.mpi_broadcast(contractions, comm=cnfg_comm)
        end

        # Store correlator entries
        if my_cnfg_rank == 0
            # Number of Momentum combinations
            Iₚ_len = length(Iₚ_chunk_arr[mom_chunk_idx])

            if direction == "forward/backward"
                # Backward direction
                for iₜ in 1:nₜ-1
                    C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[iₜ, 2, :, :, :, :, 1:Iₚ_len] = corr_arr[iₜ][1]
                    C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[iₜ, 2, :, :, :, :, 1:Iₚ_len] = corr_arr[iₜ][2]
                end

                # Source time
                C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[nₜ, 2, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][1]
                C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[nₜ, 2, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][2]
                C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[1, 1, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][1]
                C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[1, 1, :, :, :, :, 1:Iₚ_len] = corr_arr[nₜ][2]

                # Forward direction
                for iₜ in 2:nₜ
                    C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[iₜ, 1, :, :, :, :, 1:Iₚ_len] = corr_arr[iₜ+nₜ-1][1]
                    C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[iₜ, 1, :, :, :, :, 1:Iₚ_len] = corr_arr[iₜ+nₜ-1][2]
                end
            else
                for (iₜ, corr_t) in enumerate(corr_arr)    
                    C_ūcd̄c_c̄uc̄d_tdnmn̄m̄Iₚ[iₜ, 1, :, :, :, :, 1:Iₚ_len] = corr_t[1]
                    C_ūcd̄c_c̄dc̄u_tdnmn̄m̄Iₚ[iₜ, 1, :, :, :, :, 1:Iₚ_len] = corr_t[2]
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

                for mom_chunk_idx in eachindex(Iₚ_chunk_arr)
                    println("    Momenta chunk: $mom_chunk_idx of $(length(Iₚ_chunk_arr))")

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
