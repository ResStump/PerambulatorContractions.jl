# %%########################################################################################
# dad-DD_local-nonlocal_mixed_distributed.jl
#
# Compute mixed local-nonlocal diquark-antidiquark-DD correlators from perambulators, mode
# doublets and sparse modes where the contractions are done in parallel using MPI.jl.
#
# Usage:
#   dad-DD_local-nonlocal_mixed_distributed.jl -i <parms file> --nranks-per-cnfg <n>
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
# for the DD operators
Γ_DD_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ_DD = length(Γ_DD_arr)
Γ_DD_labels = ["gamma_5", "gamma_1", "gamma_2", "gamma_3", "-i1"]
# and for the dad operators
Γ₁_dad_arr = [PC.γ[1], PC.γ[2], PC.γ[3]]
Γ₂_dad_arr = [PC.γ[5]]
Nᵧ_1_dad = length(Γ₁_dad_arr)
Nᵧ_2_dad = length(Γ₂_dad_arr)
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


# %%###################################
# Momentum Pairs for Nonlocal Operators
#######################################

# Array of square of total angular momentas
Ptot_sq_arr = PC.parms_toml["Momenta nonlocal"]["Ptot_sq"]

# Maximal sum of squares of the momentum pairs that are used
p_sq_sum_max_arr = PC.parms_toml["Momenta nonlocal"]["p_sq_sum_max"]

# Compute all (relevant) momentum index pairs
Iₚ_nonlocal_arr = []
Ptot_arr = Vector{Int}[]
for (Ptot_sq, p_sq_sum_max) in zip(Ptot_sq_arr, p_sq_sum_max_arr)
    Iₚ_arr, Ptot_arr_ = PC.generate_momentum_pairs(Ptot_sq, p_sq_sum_max, ret_Ptot=true)
    append!(Iₚ_nonlocal_arr, Iₚ_arr)
    append!(Ptot_arr, Ptot_arr_)
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
mode_doublets_file(n_cnfg) = PC.parms.mode_doublets_dir/
    "mode_doublets_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"
sparse_modes_file(n_cnfg) = PC.parms.sparse_modes_dir/
    "sparse_modes_$(PC.parms_toml["Run name"]["name"])n$(n_cnfg)"

function write_correlator(n_cnfg, t₀)
    file_path = PC.parms.result_dir/"correlators_dad-DD_local-nonlocal_mixed_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    file = HDF5.h5open(string(file_path), "w")

    # Loop over all momentum index pairs for the nonlocal operator
    for (iₚ_nonlocal, Iₚ_nonlocal) in enumerate(Iₚ_nonlocal_arr)
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
        file[group_nloc_loc] = 
            C_nonlocal_local_tnmn̄m̄iₚIₚ[:, :, :, :, :, 1, iₚ_nonlocal]
        HDF5.attrs(file[group_nloc_loc])["DIMENSION_LABELS"] = 
            labels_nonlocal_local
        file[group_loc_nloc] = 
            C_local_nonlocal_tnmn̄m̄iₚIₚ[:, :, :, :, :, 1, iₚ_nonlocal]
        HDF5.attrs(file[group_loc_nloc])["DIMENSION_LABELS"] = 
            labels_local_nonlocal
    end

    # Write spin structure
    file["Spin Structure/Gamma_DD_1"] = Γ_DD_labels
    file["Spin Structure/Gamma_DD_2"] = Γ_DD_labels
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

    # Perambulators, mode doublets and sparse modes arrays
    τ_αkβlt = PC.allocate_perambulator()
    τ_charm_αkβlt = PC.allocate_perambulator()
    Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))
    sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

    # Correlators and its labels
    C_nonlocal_local_tnmn̄m̄iₚIₚ = Array{ComplexF64}(
        undef,
        PC.parms.Nₜ, Nᵧ_DD, Nᵧ_DD, Nᵧ_1_dad, Nᵧ_2_dad, 1, length(Iₚ_nonlocal_arr)
    )
    C_local_nonlocal_tnmn̄m̄iₚIₚ = Array{ComplexF64}(
        undef,
        PC.parms.Nₜ, Nᵧ_1_dad, Nᵧ_2_dad, Nᵧ_DD, Nᵧ_DD, 1, length(Iₚ_nonlocal_arr)
    )
    # Reversed order in Julia
    labels_nonlocal_local = ["CGamma2 barC", "CGamma1 barC", "CGamma2", "CGamma1", "t"]
    labels_local_nonlocal = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "t"]
end


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    @time "      DD-dad nonlocal-locals mixed contractons" begin
        if my_cnfg_rank == 0
            # Index for source time `t₀`
            i_t₀ = t₀+1

            # Unpack sparse modes arrays
            x_sink_μiₓt, x_src_μiₓt, v_sink_ciₓkt, v_src_ciₓkt = sparse_modes_arrays

            # Convert arrays to vectors of arrays in the time axis
            τ_charm_arr = eachslice(τ_charm_αkβlt, dims=5)
            τ_arr = eachslice(τ_αkβlt, dims=5)
            Φ_arr = eachslice(Φ_kltiₚ, dims=3)
            x_sink_arr = eachslice(x_sink_μiₓt, dims=3)
            v_sink_arr = eachslice(v_sink_ciₓkt, dims=4)

            # Select source time `t₀`
            Φ_kliₚ_t₀ = @view Φ_kltiₚ[:, :, i_t₀, :]
            x_src_μiₓ_t₀ = @view x_src_μiₓt[:, :, i_t₀]
            v_src_ciₓk_t₀ = @view v_src_ciₓkt[:, :, :, i_t₀]
        end

        # Function to compute contractions
        contractions = (τ_charm, τ, Φ, Φ_t₀, x_sink, x_src, v_sink, v_src) -> begin
            C_nl_arr = []
            C_ln_arr = []

            for Iₚ_nonlocal in Iₚ_nonlocal_arr
                p₁, p₂ = PC.parms.p_arr[Iₚ_nonlocal]
                Ptot = p₁ + p₂

                # Contraction for correlator of form 
                # <O_nonlocal O_local^†> and <O_local O_nonlocal^†>
                C_nl, C_ln = PC.DD_dad_nonlocal_local_mixed_contractons(
                    τ_charm, τ, Φ, Φ_t₀, (x_sink, x_src, v_sink, v_src),
                    Γ₁_dad_arr, Γ₂_dad_arr, Γ_DD_arr, Iₚ_nonlocal, [Ptot]
                )
                push!(C_nl_arr, C_nl)
                push!(C_ln_arr, C_ln)
            end

            # Return as contiguous arrays
            return stack(C_nl_arr), stack(C_ln_arr)
        end

        # Distribute workload and compute contractions
        if my_cnfg_rank == 0
            corr_arr = PC.mpi_broadcast(
                contractions, τ_charm_arr, τ_arr, Φ_arr, [Φ_kliₚ_t₀],
                x_sink_arr, [x_src_μiₓ_t₀], v_sink_arr, [v_src_ciₓk_t₀], comm=cnfg_comm
            )
        else
            PC.mpi_broadcast(contractions, comm=cnfg_comm)
        end

        # Store correlator entries
        if my_cnfg_rank == 0
            for iₜ in 1:PC.parms.Nₜ
                # Time index for storing correlator entry
                i_Δt = mod1(iₜ-t₀, PC.parms.Nₜ)
    
                C_nonlocal_local_tnmn̄m̄iₚIₚ[i_Δt, :, :, :, :, :, :] = corr_arr[iₜ][1]
                C_local_nonlocal_tnmn̄m̄iₚIₚ[i_Δt, :, :, :, :, :, :] = corr_arr[iₜ][2]
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

                compute_contractions!(t₀)
                
                # Write Correlator
                if my_cnfg_rank == 0
                    @time "    Write correlator" begin
                        #write_correlator(n_cnfg, t₀)
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
