# %%########################################################################################
# DD_mixed.jl
#
# Compute mixed local-nonlocal DD correlators from perambulators, mode doublets and
# sparse modes.
#
# Usage:
#   DD_mixed.jl -i <parms file>
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

# Array of (monomial of) γ-matrices
Γ_arr = [PC.γ[5], PC.γ[1], PC.γ[2], PC.γ[3], im*PC.γ[1]^0]
Nᵧ = length(Γ_arr)

# Continuation run?
finished_cnfgs_file = PC.parms.result_dir/"finished_cnfgs_$(myrank).txt"
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
    append!(Iₚ_nonlocal_arr, PC.generate_momentum_pairs(Ptot_sq, p_sq_sum_max))
    append!(Ptot_arr, Ptot_arr_)
end


# %%########################
# Momenta for Local Operator
############################

# The momenta for the local operator are the total momenta
p_local_arr = Ptot_arr

# Get momentum indices in mode doublets corresponding to the momentas in p_local_arr
iₚ_local_arr = [findfirst(p_ -> p_ == p, PC.parms.p_arr) for p in p_local_arr]
if any(isnothing.(iₚ_local_arr))
    throw(DomainError("a chosen momentum `p` for the local operator is not contained in " *
                      "the mode doublets."))
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
    file_path = PC.parms.result_dir/"correlators_DD_mixed_" *
        "$(PC.parms_toml["Run name"]["name"])_$(PC.parms.N_modes)modes_" *
        "n$(n_cnfg)_tsrc$(t₀).hdf5"
    hdf5_file = HDF5.h5open(string(file_path), "w")

    # Loop over all momentum index pairs for the nonlocal operator
    for (iₚ_nonlocal, Iₚ_nonlocal) in enumerate(Iₚ_nonlocal_arr)
        for (iₚ_local, Ptot) in enumerate(p_local_arr)
            # Get momenta
            p₁, p₂ = PC.parms.p_arr[Iₚ_nonlocal]
            @assert Ptot == p₁ + p₂

            # Paths to groups in hdf5 file
            Ptot_str = join(Ptot, ",")
            p₁_str = join(p₁, ",")
            group_nloc_loc =
                "Correlators/Ptot$(Ptot_str)/p_nonlocal1_$(p₁_str)/nonlocal-local"
            group_loc_nloc =
                "Correlators/Ptot$(Ptot_str)/p_nonlocal1_$(p₁_str)/local-nonlocal"

            # Write correlators with dimension labels
            hdf5_file[group_nloc_loc] = 
                C_nonlocal_local_tnmn̄m̄iₚIₚ[:, :, :, :, :, iₚ_local, iₚ_nonlocal]
            HDF5.attrs(hdf5_file[group_nloc_loc])["DIMENSION_LABELS"] = labels
            hdf5_file[group_loc_nloc] = 
                C_local_nonlocal_tnmn̄m̄iₚIₚ[:, :, :, :, :, iₚ_local, iₚ_nonlocal]
            HDF5.attrs(hdf5_file[group_loc_nloc])["DIMENSION_LABELS"] = labels
        end
    end

    # Write parameter file and program information
    hdf5_file["parms.toml"] = PC.parms.parms_toml_string
    hdf5_file["Program Information"] = PC.parms_toml["Program Information"]

    close(hdf5_file)

    return
end


# %%#############
# Allocate Arrays
#################

# Select valid cnfg number
n_cnfg = PC.parms.cnfg_indices[1]

# Perambulators, mode doublets and sparse modes arrays
τ_αkβlt = PC.allocate_perambulator()
τ_charm_αkβlt = PC.allocate_perambulator()
Φ_kltiₚ = PC.allocate_mode_doublets(mode_doublets_file(n_cnfg))
sparse_modes_arrays = PC.allocate_sparse_modes(sparse_modes_file(n_cnfg))

# Correlator and its labels
correlator_size = (PC.parms.Nₜ, Nᵧ, Nᵧ, Nᵧ, Nᵧ, length(p_local_arr), length(Iₚ_nonlocal_arr))
C_nonlocal_local_tnmn̄m̄iₚIₚ = Array{ComplexF64}(undef, correlator_size)
C_local_nonlocal_tnmn̄m̄iₚIₚ = Array{ComplexF64}(undef, correlator_size)
# Reversed order in Julia
labels = ["Gamma2 bar", "Gamma1 bar", "Gamma2", "Gamma1", "t"]


# %%#########
# Computation
#############

function compute_contractions!(t₀)
    # Loop over all momentum index combinations
    for (iₚ_nonlocal, Iₚ_nonlocal) in enumerate(Iₚ_nonlocal_arr)
        println("    Momenta for nonlocal operator: $(PC.parms.p_arr[Iₚ_nonlocal])")
        
        @time "      DD mixed contractons" begin
            # Contraction for correlator of form 
            # <O_nonlocal O_local^†> and <O_local O_nonlocal^†>
            C_nonlocal_local_tnmn̄m̄iₚ_Iₚ =
                @view C_nonlocal_local_tnmn̄m̄iₚIₚ[:, :, :, :, :, :, iₚ_nonlocal]
            C_local_nonlocal_tnmn̄m̄iₚ_Iₚ =
                @view C_local_nonlocal_tnmn̄m̄iₚIₚ[:, :, :, :, :, :, iₚ_nonlocal]
            PC.DD_mixed_contractons!(
                C_nonlocal_local_tnmn̄m̄iₚ_Iₚ, C_local_nonlocal_tnmn̄m̄iₚ_Iₚ,
                τ_charm_αkβlt, τ_αkβlt, Φ_kltiₚ, sparse_modes_arrays, Γ_arr, t₀,
                Iₚ_nonlocal, p_local_arr
            )
        end
        println()
    end
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
